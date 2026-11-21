// Ops on GPU tensors: kernels compiled at runtime by NVRTC. See ml/cuda.hpp.
//
// Every GPU op computes its forward with a kernel and attaches a
// backward_graph -- its gradient written with ops, which run on the GPU too.
// The autograd engine runs that graph for any node on the GPU, and the
// gradients accumulate on the device. There are no raw backward kernels to
// keep in step with the forward, and the gradients are differentiable again.
#include "cuda_ops_common.hpp"
#include "ml/cuda.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <mutex>

using namespace std;
using namespace cuda_common;

namespace cuda_common {

// The device-side helpers every kernel source starts with: dtype codes in
// DType's order, loads and stores for each (float16 / bfloat16 converted by
// hand, so no CUDA header is needed), and the strided index map.
const char* const COMMON = R"(
#define MAX_DIMS 8
struct Index { long long shape[MAX_DIMS]; long long st[MAX_DIMS]; int nd; };
// Float32, Float64, Float16, BFloat16, Int64, Int32, Bool -- DType's order
__device__ float h2f(unsigned short h) {
    unsigned int s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
    if (e == 0) {
        if (m == 0) f = s;
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; f = s | (e << 23) | (m << 13); }
    } else if (e == 31) f = s | 0x7f800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    return __int_as_float((int)f);
}
__device__ unsigned short f2h(float x) {
    unsigned int u = (unsigned int)__float_as_int(x), s = (u >> 16) & 0x8000u;
    int e = (int)((u >> 23) & 0xff) - 127 + 15;
    unsigned int m = u & 0x7fffff;
    if (((u >> 23) & 0xff) == 0xff) return (unsigned short)(s | 0x7c00u | (m ? 0x200u : 0u));
    if (e >= 31) return (unsigned short)(s | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (unsigned short)s;
        m |= 0x800000u;
        unsigned int shift = (unsigned int)(14 - e), r = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1u))) r++;
        return (unsigned short)(s | r);
    }
    unsigned int r = ((unsigned int)e << 10) | (m >> 13), rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1u))) r++;
    return (unsigned short)(s | r);
}
__device__ float bf2f(unsigned short b) { return __int_as_float((int)((unsigned int)b << 16)); }
__device__ unsigned short f2bf(float x) {
    unsigned int u = (unsigned int)__float_as_int(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
__device__ double ld(const void* p, int dt, long long i) {
    switch (dt) {
        case 0: return ((const float*)p)[i];
        case 1: return ((const double*)p)[i];
        case 2: return h2f(((const unsigned short*)p)[i]);
        case 3: return bf2f(((const unsigned short*)p)[i]);
        case 4: return (double)((const long long*)p)[i];
        case 5: return ((const int*)p)[i];
        default: return ((const unsigned char*)p)[i] ? 1.0 : 0.0;
    }
}
__device__ long long ldi(const void* p, int dt, long long i) {
    switch (dt) {
        case 4: return ((const long long*)p)[i];
        case 5: return ((const int*)p)[i];
        case 6: return ((const unsigned char*)p)[i] ? 1 : 0;
        default: return (long long)ld(p, dt, i);
    }
}
__device__ void st(void* p, int dt, long long i, double v) {
    switch (dt) {
        case 0: ((float*)p)[i] = (float)v; break;
        case 1: ((double*)p)[i] = v; break;
        case 2: ((unsigned short*)p)[i] = f2h((float)v); break;
        case 3: ((unsigned short*)p)[i] = f2bf((float)v); break;
        case 4: ((long long*)p)[i] = (long long)v; break;
        case 5: ((int*)p)[i] = (int)v; break;
        default: ((unsigned char*)p)[i] = v != 0.0; break;
    }
}
__device__ void sti(void* p, int dt, long long i, long long v) {
    switch (dt) {
        case 4: ((long long*)p)[i] = v; break;
        case 5: ((int*)p)[i] = (int)v; break;
        case 6: ((unsigned char*)p)[i] = v != 0; break;
        default: st(p, dt, i, (double)v);
    }
}
// single precision where no float64 / int64 is involved: consumer GPUs run
// double at 1/32 the rate
__device__ float ldf(const void* p, int dt, long long i) {
    switch (dt) {
        case 0: return ((const float*)p)[i];
        case 2: return h2f(((const unsigned short*)p)[i]);
        case 3: return bf2f(((const unsigned short*)p)[i]);
        case 5: return (float)((const int*)p)[i];
        case 6: return ((const unsigned char*)p)[i] ? 1.0f : 0.0f;
        default: return (float)ld(p, dt, i);
    }
}
__device__ void stf(void* p, int dt, long long i, float v) {
    switch (dt) {
        case 0: ((float*)p)[i] = v; break;
        case 2: ((unsigned short*)p)[i] = f2h(v); break;
        case 3: ((unsigned short*)p)[i] = f2bf(v); break;
        case 6: ((unsigned char*)p)[i] = v != 0.0f; break;
        default: st(p, dt, i, (double)v);
    }
}
// a fused kernel's activations: io 0 float32, 2 float16, 3 bfloat16 (uniform
// per launch, so the branch costs nothing)
__device__ __forceinline__ float ldx(const void* p, int io, long long i) {
    return io == 0 ? ((const float*)p)[i]
                   : io == 3 ? bf2f(((const unsigned short*)p)[i]) : h2f(((const unsigned short*)p)[i]);
}
__device__ __forceinline__ void stx(void* p, int io, long long i, float v) {
    if (io == 0) ((float*)p)[i] = v;
    else ((unsigned short*)p)[i] = io == 3 ? f2bf(v) : f2h(v);
}
// element i of the output, read through a strided / broadcast operand
__device__ long long at(const Index& ix, long long i) {
    if (ix.nd < 0) return i;                     // contiguous, same shape
    long long r = i, o = 0;
    for (int d = ix.nd - 1; d >= 0; d--) { long long q = r % ix.shape[d]; r /= ix.shape[d]; o += q * ix.st[d]; }
    return o;
}
__device__ long long gid() { return (long long)blockIdx.x * blockDim.x + threadIdx.x; }
)";

// t's strides laid against `out`: its own, 0 where broadcast or missing
Index index_for(const TensorPtr& t, const Shape& out) {
    if ((int)out.size() > MAX_DIMS)
        throw runtime_error("cuda: more than 8 dimensions");
    Index ix{};
    ix.nd = (int)out.size();
    const int off = ix.nd - t->dim();
    for (int d = 0; d < ix.nd; d++) {
        ix.shape[d] = out[d];
        if (d < off) { ix.st[d] = 0; continue; }
        const int k = d - off;
        ix.st[d] = t->shape()[k] == 1 ? 0 : t->strides()[k];
    }
    // Merge dimensions the walk crosses as one: d into d+1 when stepping off
    // the end of d+1 lands where the next d would (broadcast runs merge too).
    // `at` pays two 64-bit divisions per dimension, which on GeForce cards
    // made a ReLU three times slower than a copy; a contiguous operand ends
    // as nd = -1 and indexes with no division at all.
    Index m{};
    m.nd = 0;
    for (int d = 0; d < ix.nd; d++) {
        if (ix.shape[d] == 1) continue;
        if (m.nd > 0 && m.st[m.nd - 1] == ix.st[d] * ix.shape[d]) {
            m.shape[m.nd - 1] *= ix.shape[d];
            m.st[m.nd - 1] = ix.st[d];
            continue;
        }
        m.shape[m.nd] = ix.shape[d];
        m.st[m.nd] = ix.st[d];
        m.nd++;
    }
    if (m.nd == 0 || (m.nd == 1 && m.st[0] == 1)) m.nd = -1;
    return m;
}

string src(const char* body) { return string(COMMON) + body; }

// kernel sources are built once and live for the process
const string& UNARY_SRC() {
    static const string s = src(R"(
// NAME: any layout; NAME##4: contiguous, four elements a thread through
// 16-byte loads and stores (n4 = n / 4, the buffers 16-byte aligned)
#define UNARY(NAME, EXPR) extern "C" __global__ void NAME(const float* a, float* y, long long n, Index ix, float p0, float p1) { \
    long long i = gid(); if (i >= n) return; float x = a[at(ix, i)]; y[i] = EXPR; } \
extern "C" __global__ void NAME##4(const float4* a, float4* y, long long n4, float p0, float p1) { \
    long long i = gid(); if (i >= n4) return; const float4 v = a[i]; float4 r; float x; \
    x = v.x; r.x = EXPR; x = v.y; r.y = EXPR; x = v.z; r.z = EXPR; x = v.w; r.w = EXPR; y[i] = r; }
UNARY(u_sadd, x + p0)
UNARY(u_smul, x * p0)
UNARY(u_rsub, p0 - x)
UNARY(u_ssub, x - p0)
UNARY(u_rdiv, p0 / x)
UNARY(u_sdiv, x / p0)
UNARY(u_exp, expf(x))
UNARY(u_log, logf(x))
UNARY(u_sqrt, sqrtf(x))
UNARY(u_abs, fabsf(x))
UNARY(u_relu, x > 0.0f ? x : 0.0f)
UNARY(u_sigmoid, 1.0f / (1.0f + expf(-x)))
UNARY(u_tanh, tanhf(x))
UNARY(u_gelu, 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))))
UNARY(u_pow, powf(x, p0))
UNARY(u_clip, fminf(fmaxf(x, p0), p1))
)");
    return s;
}

const string& BINARY_SRC() {
    static const string s = src(R"(
#define BINARY(NAME, EXPR) extern "C" __global__ void NAME(const float* a, const float* b, float* c, long long n, Index ia, Index ib) { \
    long long i = gid(); if (i >= n) return; float x = a[at(ia, i)], y = b[at(ib, i)]; c[i] = EXPR; } \
extern "C" __global__ void NAME##4(const float4* a, const float4* b, float4* c, long long n4) { \
    long long i = gid(); if (i >= n4) return; const float4 u = a[i], w = b[i]; float4 r; float x, y; \
    x = u.x; y = w.x; r.x = EXPR; x = u.y; y = w.y; r.y = EXPR; x = u.z; y = w.z; r.z = EXPR; \
    x = u.w; y = w.w; r.w = EXPR; c[i] = r; }
BINARY(b_add, x + y)
BINARY(b_sub, x - y)
BINARY(b_mul, x * y)
BINARY(b_div, x / y)
)");
    return s;
}

// comparisons and logic on any dtypes, bool out
const string& COMPARE_SRC() {
    static const string s = src(R"(
extern "C" __global__ void c_compare(const void* a, int da, const void* b, int db, unsigned char* y,
                                     long long n, Index ia, Index ib, int op, int integral) {
    long long i = gid(); if (i >= n) return;
    long long oa = at(ia, i), ob = at(ib, i);
    bool r;
    if (op >= 6) {
        bool p = ld(a, da, oa) != 0.0, q = ld(b, db, ob) != 0.0;
        r = op == 6 ? (p && q) : op == 7 ? (p || q) : (p != q);
    } else if (integral) {
        long long x = ldi(a, da, oa), z = ldi(b, db, ob);
        r = op == 0 ? x == z : op == 1 ? x != z : op == 2 ? x < z : op == 3 ? x <= z : op == 4 ? x > z : x >= z;
    } else if (da != 1 && db != 1 && da != 4 && db != 4) {
        float x = ldf(a, da, oa), z = ldf(b, db, ob);
        r = op == 0 ? x == z : op == 1 ? x != z : op == 2 ? x < z : op == 3 ? x <= z : op == 4 ? x > z : x >= z;
    } else {
        double x = ld(a, da, oa), z = ld(b, db, ob);
        r = op == 0 ? x == z : op == 1 ? x != z : op == 2 ? x < z : op == 3 ? x <= z : op == 4 ? x > z : x >= z;
    }
    y[i] = r;
}
)");
    return s;
}

// any dtype to any dtype, strided in, contiguous out; also the plain copy
const string& CAST_SRC() {
    static const string s = src(R"(
extern "C" __global__ void k_cast(const void* a, int da, void* y, int dy, long long n, Index ix) {
    long long i = gid(); if (i >= n) return;
    long long o = at(ix, i);
    if (da == dy && (da == 0 || da == 5)) { ((int*)y)[i] = ((const int*)a)[o]; return; }   // 4-byte copy
    if (da >= 4 && dy >= 4) sti(y, dy, i, ldi(a, da, o));
    else if (da == 1 || dy == 1 || da == 4 || dy == 4) st(y, dy, i, ld(a, da, o));        // float64 / int64: exact
    else stf(y, dy, i, ldf(a, da, o));
}
// grad += a, or grad = a (assign: a tensor's first gradient, into a buffer
// that was never cleared). Loads in single precision: ld went through double,
// 1/32 rate on GeForce cards.
extern "C" __global__ void k_accumulate(float* g, const void* a, int da, long long n, Index ix, int assign) {
    long long i = gid(); if (i >= n) return;
    const float v = ldf(a, da, at(ix, i));
    g[i] = assign ? v : g[i] + v;
}
// float32 <-> bfloat16 / float16, contiguous, four at a time: the casts around
// mixed-precision gemms (weights in, gradients back) -- the generic k_cast
// switched on both dtypes per element and ran at a fraction of bandwidth
extern "C" __global__ void k_f32_to_half4(const float4* a, uint2* y, long long n4, int dy) {
    long long i = gid(); if (i >= n4) return;
    const float4 v = a[i];
    unsigned short h0, h1, h2, h3;
    if (dy == 3) { h0 = f2bf(v.x); h1 = f2bf(v.y); h2 = f2bf(v.z); h3 = f2bf(v.w); }
    else { h0 = f2h(v.x); h1 = f2h(v.y); h2 = f2h(v.z); h3 = f2h(v.w); }
    y[i] = make_uint2((unsigned)h0 | ((unsigned)h1 << 16), (unsigned)h2 | ((unsigned)h3 << 16));
}
extern "C" __global__ void k_half_to_f324(const uint2* a, float4* y, long long n4, int da) {
    long long i = gid(); if (i >= n4) return;
    const uint2 v = a[i];
    const unsigned short h0 = v.x & 0xffff, h1 = v.x >> 16, h2 = v.y & 0xffff, h3 = v.y >> 16;
    y[i] = da == 3 ? make_float4(bf2f(h0), bf2f(h1), bf2f(h2), bf2f(h3))
                   : make_float4(h2f(h0), h2f(h1), h2f(h2), h2f(h3));
}
// float32 gradients, contiguous, four at a time
extern "C" __global__ void k_accumulate4(float4* g, const float4* a, long long n4, int assign) {
    long long i = gid(); if (i >= n4) return;
    const float4 v = a[i];
    if (assign) { g[i] = v; return; }
    float4 o = g[i];
    o.x += v.x; o.y += v.y; o.z += v.z; o.w += v.w;
    g[i] = o;
}
extern "C" __global__ void k_fill(float* y, long long n, float v) {
    long long i = gid(); if (i >= n) return;
    y[i] = v;
}
extern "C" __global__ void k_where(const void* c, int dc, const float* a, const float* b, float* y, long long n,
                                   Index ic, Index ia, Index ib) {
    long long i = gid(); if (i >= n) return;
    y[i] = ld(c, dc, at(ic, i)) != 0.0 ? a[at(ia, i)] : b[at(ib, i)];
}
)");
    return s;
}

// sums: over a middle dimension of [outer, len, inner], and over everything
const string& REDUCE_SRC() {
    static const string s = src(R"(
extern "C" __global__ void r_dim(const float* a, float* y, long long outer, long long len, long long inner, float scale) {
    long long i = gid(); if (i >= outer * inner) return;
    long long o = i / inner, j = i % inner;
    const float* p = a + o * len * inner + j;
    float acc = 0.0f;
    for (long long k = 0; k < len; k++) acc += p[k * inner];
    y[i] = acc * scale;
}
// The same when there are few outputs and a long dimension -- the bias
// gradient sums 1024 rows into 256 columns, which one thread per column ran on
// 256 threads. Here each thread sums a chunk of `ch` and adds it atomically,
// so there are outer * inner * chunks threads; y must start at zero.
extern "C" __global__ void r_dim_chunk(const float* a, float* y, long long outer, long long len, long long inner,
                                      float scale, long long ch) {
    const long long outs = outer * inner, chunks = (len + ch - 1) / ch;
    long long i = gid(); if (i >= outs * chunks) return;
    const long long c = i / outs, oi = i % outs, o = oi / inner, j = oi % inner;
    const long long k0 = c * ch, k1 = min(len, k0 + ch);
    const float* p = a + o * len * inner + j;
    float acc = 0.0f;
    for (long long k = k0; k < k1; k++) acc += p[k * inner];
    atomicAdd(&y[oi], acc * scale);
}
// The same as a tiled column sum: a block of 32 x 8 threads takes 32 columns
// of 128 rows (16 per thread, stride 8), folds its 8 partials in shared
// memory and adds one value per column -- reads stay coalesced across the
// columns, and there are enough blocks to fill the GPU. y starts at zero.
// grid: (ceil(inner / 32), ceil(len / 128), outer)
extern "C" __global__ void r_cols(const float* a, float* y, long long len, long long inner, float scale) {
    __shared__ float part[8][33];
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    const long long j = (long long)blockIdx.x * 32 + tx;
    const long long r0 = (long long)blockIdx.y * 128;
    const float* p = a + (long long)blockIdx.z * len * inner;
    float acc = 0.0f;
    if (j < inner)
        for (long long r = r0 + ty; r < len && r < r0 + 128; r += 8) acc += p[r * inner + j];
    part[ty][tx] = acc;
    __syncthreads();
    if (ty == 0 && j < inner) {
        float t = 0.0f;
        for (int k = 0; k < 8; k++) t += part[k][tx];
        atomicAdd(&y[(long long)blockIdx.z * inner + j], t * scale);
    }
}
// one partial sum per block, tree-reduced in shared memory
extern "C" __global__ void r_partial(const float* a, float* part, long long n) {
    __shared__ float s[256];
    float acc = 0.0f;
    for (long long i = gid(); i < n; i += (long long)gridDim.x * blockDim.x) acc += a[i];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = s[0];
}
// softmax over rows of `cols`, one block per row
extern "C" __global__ void r_softmax(const float* a, float* y, long long cols) {
    __shared__ float s[256];
    const float* x = a + (long long)blockIdx.x * cols;
    float* out = y + (long long)blockIdx.x * cols;
    float m = -3.402823466e38f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, x[j]);
    s[threadIdx.x] = m;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + w]);
        __syncthreads();
    }
    m = s[0];
    __syncthreads();
    float t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) { float e = expf(x[j] - m); out[j] = e; t += e; }
    s[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w];
        __syncthreads();
    }
    float inv = 1.0f / s[0];
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) out[j] *= inv;
}
)");
    return s;
}

cuda::Kernel K(const char* name, const string& source) { return {name, source.c_str()}; }

TensorPtr empty_cuda(const Shape& shape, DType d) {
    return make_shared<Tensor>(shape, Tensor::Uninitialized{}, d, Device::CUDA);
}

const void* base(const TensorPtr& t) {
    return static_cast<const unsigned char*>(t->storage()->data()) + t->offset() * dtype_size(t->dtype());
}

void attach(const TensorPtr& out, vector<TensorPtr> inputs, const char* name,
            function<vector<TensorPtr>(const TensorPtr&, const TensorPtr&)> graph) {
    if (!grad_enabled()) return;              // no_grad: no node, nothing kept for a backward
    bool need = false;
    for (auto& t : inputs) need = need || (t && t->requires_grad());
    if (!need) return;
    auto node = make_node(out, std::move(inputs));
    node->name = name;
    node->backward_graph = std::move(graph);
}

} // namespace cuda_common

namespace cuda_ops {

TensorPtr softmax_backward(const TensorPtr& y, const TensorPtr& G);
TensorPtr cross_entropy_backward(const TensorPtr& x, const TensorPtr& targets, const TensorPtr& G);

// An operand on the GPU: a CPU scalar (0-d or one element, as scalar_tensor
// and wrapped Python numbers make) is moved over; a full CPU tensor is an error.
TensorPtr on_device(const TensorPtr& t, const char* op) {
    if (t->device() == Device::CUDA) return t;
    if (t->numel() == 1) return t->to(Device::CUDA);
    throw runtime_error(string(op) + ": the operands are on different devices; move both with "
                        ".to(cuda) or .to(cpu)");
}

TensorPtr cast(const TensorPtr& a, DType d) {
    auto out = empty_cuda(a->shape(), d);
    Index ix = index_for(a, a->shape());
    const void* pa = base(a);
    void* py = out->storage()->data();
    int da = (int)a->dtype(), dy = (int)d;
    long long n = out->numel();
    const bool half_d = d == DType::BFloat16 || d == DType::Float16;
    const bool half_a = a->dtype() == DType::BFloat16 || a->dtype() == DType::Float16;
    if (ix.nd < 0 && n >= 4 && n % 4 == 0 && ((uintptr_t)pa & 15) == 0 && ((uintptr_t)py & 15) == 0 &&
        ((a->dtype() == F32 && half_d) || (half_a && d == F32))) {
        long long n4 = n / 4;
        if (a->dtype() == F32) {
            void* args[] = {&pa, &py, &n4, &dy};
            cuda::launch(K("k_f32_to_half4", CAST_SRC()), n4, args);
        } else {
            void* args[] = {&pa, &py, &n4, &da};
            cuda::launch(K("k_half_to_f324", CAST_SRC()), n4, args);
        }
        return out;
    }
    void* args[] = {&pa, &da, &py, &dy, &n, &ix};
    cuda::launch(K("k_cast", CAST_SRC()), n, args);
    return out;
}

TensorPtr copy(const TensorPtr& a) { return cast(a, a->dtype()); }

// operands for a float32 kernel: on the device, as float32
// through cast_op, so a half-precision input's gradient flows back through
// the conversion
TensorPtr f32(const TensorPtr& t, const char* op) {
    auto d = on_device(t, op);
    return d->dtype() == F32 ? d : cast_op(d, F32);
}

// a node on `out` over `inputs`: `fused` for a plain backward, `reference` (the
// layer's op form) when the gradient is recorded
void attach_fused(const TensorPtr& out, vector<TensorPtr> inputs, const char* name, Reference reference,
                         function<vector<TensorPtr>(const TensorPtr& G)> fused) {
    if (!grad_enabled()) return;
    bool need = false;
    for (auto& t : inputs) need = need || (t && t->requires_grad());
    if (!need) return;
    auto node = make_node(out, inputs);
    node->name = name;
    autograd::attach_reference(*node, reference);
    auto ref_graph = node->backward_graph;
    node->backward_graph = [ref_graph, fused, inputs](const TensorPtr& o, const TensorPtr& G) -> vector<TensorPtr> {
        if (grad_enabled()) return ref_graph(o, G);
        auto parts = fused(G);
        for (size_t i = 0; i < parts.size() && i < inputs.size(); i++)
            if (!inputs[i] || !inputs[i]->requires_grad()) parts[i] = nullptr;
        return parts;
    };
}

// The backward of every element-wise op as one kernel: dx from x, y = f(x)
// and dy. A plain backward used to run each op's gradient formula as ops --
// ReLU's was a comparison, a cast and a multiply -- three launches and three
// passes over memory where one does.
static const string& UNARY_BACK_SRC() {
    static const string s = src(R"(
extern "C" __global__ void u_back(int op, const float* x, const float* y, const float* g, float* dx,
                                  long long n, float p0, float p1) {
    long long i = gid(); if (i >= n) return;
    // load only what the formula reads: `op` is the same for every thread, so
    // skipped loads cost nothing (ReLU reads x and g, not the output)
    const bool use_y = op == 5 || op == 7 || op == 10 || op == 11;
    const bool use_x = op == 3 || op == 6 || op == 8 || op == 9 || op >= 12;
    const float v = use_x ? x[i] : 0.0f, o = use_y ? y[i] : 0.0f, d = g[i];
    float r;
    switch (op) {
        case 0:  r = d; break;                                           // x + s, x - s
        case 1:  r = d * p0; break;                                      // s * x
        case 2:  r = -d; break;                                          // s - x
        case 3:  r = -d * p0 / (v * v); break;                           // s / x
        case 4:  r = d / p0; break;                                      // x / s
        case 5:  r = d * o; break;                                       // exp
        case 6:  r = d / v; break;                                       // log
        case 7:  r = d / (2.0f * o); break;                              // sqrt
        case 8:  r = v > 0.0f ? d : (v < 0.0f ? -d : 0.0f); break;       // abs
        case 9:  r = v > 0.0f ? d : 0.0f; break;                         // relu
        case 10: r = d * o * (1.0f - o); break;                          // sigmoid
        case 11: r = d * (1.0f - o * o); break;                          // tanh
        case 12: r = d * p0 * powf(v, p0 - 1.0f); break;                 // pow
        default: r = (v > p0 && v < p1) ? d : 0.0f; break;               // clip
    }
    dx[i] = r;
}
)");
    return s;
}

static int unary_backward_code(const char* kernel) {
    static const char* const names[] = {"u_sadd", "u_smul", "u_rsub", "u_rdiv", "u_sdiv", "u_exp", "u_log", "u_sqrt",
                                        "u_abs", "u_relu", "u_sigmoid", "u_tanh", "u_pow", "u_clip"};
    static const int codes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    const string k = kernel;
    if (k == "u_ssub") return 0;
    for (int i = 0; i < 14; i++) if (k == names[i]) return codes[i];
    return -1;
}

static TensorPtr unary_backward(int op, const TensorPtr& x0, const TensorPtr& y0, const TensorPtr& G0, float p0, float p1) {
    auto cont = [](const TensorPtr& t) { return t->is_contiguous() ? t : contiguous(t); };
    auto x = cont(x0), y = cont(y0);
    auto G = cont(G0->dtype() == F32 ? G0 : cast_op(G0, F32));
    auto dx = empty_cuda(x->shape());
    const void* px = base(x);
    const void* py = base(y);
    const void* pg = base(G);
    void* pd = dx->storage()->data();
    long long n = x->numel();
    int o = op;
    float a = p0, b = p1;
    void* args[] = {&o, &px, &py, &pg, &pd, &n, &a, &b};
    cuda::launch(K("u_back", UNARY_BACK_SRC()), n, args);
    return dx;
}

// The four-at-a-time kernels apply: contiguous operands (nd < 0), a count
// divisible by four, and 16-byte aligned buffers.
static bool vec4(long long n, const Index& ix, const void* a, const void* y) {
    return n >= 4 && n % 4 == 0 && ix.nd < 0 && ((uintptr_t)a & 15) == 0 && ((uintptr_t)y & 15) == 0;
}
// "u_exp" -> "u_exp4", names kept for the process (Kernel holds the pointer)
static const char* kernel4_name(const char* kernel) {
    static std::mutex mu;
    static auto& names = *new std::unordered_map<string, string>();
    std::lock_guard<std::mutex> lock(mu);
    auto it = names.find(kernel);
    if (it == names.end()) it = names.emplace(kernel, string(kernel) + "4").first;
    return it->second.c_str();
}

TensorPtr unary(const char* kernel, const char* name, const TensorPtr& a0, float p0, float p1,
                dispatch::Graph1 graph) {
    if (a0->dtype() != F32 && is_floating(a0->dtype()) && a0->dtype() != DType::Bool &&
        a0->dtype() != DType::Float16 && a0->dtype() != DType::BFloat16)
        throw runtime_error(string(name) + ": on the GPU only float32 (and half) so far, got " +
                            dtype_name(a0->dtype()));
    auto a = f32(a0, name);
    auto out = empty_cuda(a->shape());
    Index ix = index_for(a, a->shape());
    const void* pa = base(a);
    void* py = out->storage()->data();
    long long n = out->numel();
    if (vec4(n, ix, pa, py)) {
        long long n4 = n / 4;
        void* args[] = {&pa, &py, &n4, &p0, &p1};
        cuda::launch(K(kernel4_name(kernel), UNARY_SRC()), n4, args);
    } else {
        void* args[] = {&pa, &py, &n, &ix, &p0, &p1};
        cuda::launch(K(kernel, UNARY_SRC()), n, args);
    }
    const bool is_gelu = string(kernel) == "u_gelu";
    const int back = unary_backward_code(kernel);
    attach(out, {a}, name, [a, graph, is_gelu, back, p0, p1](const TensorPtr& out, const TensorPtr& G) -> vector<TensorPtr> {
        if (!grad_enabled()) {
            if (is_gelu) return {gelu_backward(a, G)};
            if (back >= 0) return {unary_backward(back, a, out, G, p0, p1)};
        }
        return {graph(a, out, G)};
    });
    return out;
}

TensorPtr binary(int op, const char* name, const TensorPtr& a0, const TensorPtr& b0) {
    static const char* const names[] = {"b_add", "b_sub", "b_mul", "b_div"};
    const DType d = result_type(a0, b0);
    if (d == DType::Float64)
        throw runtime_error(string(name) + ": float64 is not supported on the GPU yet");
    auto a = f32(a0, name), b = f32(b0, name);
    const Shape shape = broadcast_shapes(a->shape(), b->shape());
    auto out = empty_cuda(shape);
    Index ia = index_for(a, shape), ib = index_for(b, shape);
    const void* pa = base(a);
    const void* pb = base(b);
    void* pc = out->storage()->data();
    long long n = out->numel();
    if (vec4(n, ia, pa, pc) && vec4(n, ib, pb, pc)) {
        static const char* const names4[] = {"b_add4", "b_sub4", "b_mul4", "b_div4"};
        long long n4 = n / 4;
        void* args[] = {&pa, &pb, &pc, &n4};
        cuda::launch(K(names4[op], BINARY_SRC()), n4, args);
    } else {
        void* args[] = {&pa, &pb, &pc, &n, &ia, &ib};
        cuda::launch(K(names[op], BINARY_SRC()), n, args);
    }
    attach(out, {a, b}, name, [a, b, op](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        TensorPtr ga, gb;
        switch (op) {
            case 0: ga = G; gb = G; break;
            case 1: ga = G; gb = multiply(-1.0f, G); break;
            case 2: ga = multiply(G, b); gb = multiply(G, a); break;
            default: ga = divide(G, b); gb = multiply(-1.0f, divide(multiply(G, a), multiply(b, b))); break;
        }
        return {a->requires_grad() ? sum_to(ga, a->shape()) : nullptr,
                b->requires_grad() ? sum_to(gb, b->shape()) : nullptr};
    });
    // integer and bool results are float32 for now; half results keep their dtype
    if (d == DType::Float16 || d == DType::BFloat16) return cast_op(out, d);
    return out;
}

TensorPtr compare(int op, const TensorPtr& a0, const TensorPtr& b0) {
    TensorPtr bb = b0;
    if (b0->numel() == 1 && b0->dtype() == DType::Float64 && a0->dtype() == F32 && b0->device() == Device::CPU) {
        const double v = b0->item_double();
        if ((double)(float)v == v) bb = scalar_tensor(v, F32);
    }
    auto a = on_device(a0, "compare"), b = on_device(bb, "compare");
    const Shape shape = broadcast_shapes(a->shape(), b->shape());
    auto out = empty_cuda(shape, DType::Bool);
    Index ia = index_for(a, shape), ib = index_for(b, shape);
    const void* pa = base(a);
    const void* pb = base(b);
    void* py = out->storage()->data();
    int da = (int)a->dtype(), db = (int)b->dtype();
    int integral = !is_floating(a->dtype()) && !is_floating(b->dtype());
    long long n = out->numel();
    void* args[] = {&pa, &da, &pb, &db, &py, &n, &ia, &ib, &op, &integral};
    cuda::launch(K("c_compare", COMPARE_SRC()), n, args);
    return out;
}

TensorPtr where(const TensorPtr& c0, const TensorPtr& a0, const TensorPtr& b0) {
    auto c = on_device(c0, "where"), a = f32(a0, "where"), b = f32(b0, "where");
    const Shape shape = broadcast_shapes(broadcast_shapes(c->shape(), a->shape()), b->shape());
    auto out = empty_cuda(shape);
    Index ic = index_for(c, shape), ia = index_for(a, shape), ib = index_for(b, shape);
    const void* pc = base(c);
    const void* pa = base(a);
    const void* pb = base(b);
    void* py = out->storage()->data();
    int dc = (int)c->dtype();
    long long n = out->numel();
    void* args[] = {&pc, &dc, &pa, &pb, &py, &n, &ic, &ia, &ib};
    cuda::launch(K("k_where", CAST_SRC()), n, args);
    attach(out, {a, b}, "where", [c, a, b](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        auto zero = scalar_tensor(0.0, DType::Float32);
        return {a->requires_grad() ? sum_to(::where(c, G, zero), a->shape()) : nullptr,
                b->requires_grad() ? sum_to(::where(c, zero, G), b->shape()) : nullptr};
    });
    return out;
}

TensorPtr reduce(const char* name, const TensorPtr& a0, int dim, bool keepdim, bool mean) {
    auto a = f32(a0, name);
    if (!a->is_contiguous()) a = contiguous(a);
    const int nd = a->dim();
    int64_t outer = 1, len = a->numel(), inner = 1;
    Shape out_shape;
    if (dim == -1) {
        out_shape = keepdim ? Shape(max(nd, 1), 1) : Shape{1};
    } else {
        if (dim < 0 || dim >= nd)
            throw runtime_error(string(name) + ": dim " + to_string(dim) + " is out of range");
        for (int k = 0; k < dim; k++) outer *= a->shape()[k];
        for (int k = dim + 1; k < nd; k++) inner *= a->shape()[k];
        len = a->shape()[dim];
        for (int k = 0; k < nd; k++)
            if (k != dim) out_shape.push_back(a->shape()[k]);
            else if (keepdim) out_shape.push_back(1);
        if (out_shape.empty()) out_shape = {1};
    }
    auto out = empty_cuda(out_shape);
    const float scale = mean ? 1.0f / (float)len : 1.0f;
    const void* pa = base(a);
    void* py = out->storage()->data();
    if (dim == -1 && len > 4096) {
        // two passes: per-block partial sums, then one block over the partials
        const long long blocks = min<long long>((len + 255) / 256, 1024);
        auto part = empty_cuda({blocks});
        void* pp = part->storage()->data();
        long long n = len;
        void* a1[] = {&pa, &pp, &n};
        cuda::launch(K("r_partial", REDUCE_SRC()), (unsigned)blocks, 1, 1, 256, 1, 1, 0, a1);
        long long one = 1, nb = blocks;
        float s = scale;
        long long o1 = 1;
        void* a2[] = {&pp, &py, &one, &nb, &o1, &s};
        cuda::launch(K("r_dim", REDUCE_SRC()), 1, a2);
    } else if (outer * inner < 16384 && len >= 64 && inner >= 32 && outer < 65536) {
        // few outputs, long sums over contiguous columns (bias gradients)
        cuda::memset_zero(py, (size_t)out->numel() * sizeof(float));
        long long l = len, in = inner;
        float s = scale;
        void* args[] = {&pa, &py, &l, &in, &s};
        cuda::launch(K("r_cols", REDUCE_SRC()), (unsigned)((inner + 31) / 32), (unsigned)((len + 127) / 128),
                     (unsigned)outer, 256, 1, 1, 0, args);
    } else if (outer * inner < 16384 && len >= 64) {
        cuda::memset_zero(py, (size_t)out->numel() * sizeof(float));
        long long o = outer, l = len, in = inner, ch = 32;
        float s = scale;
        void* args[] = {&pa, &py, &o, &l, &in, &s, &ch};
        cuda::launch(K("r_dim_chunk", REDUCE_SRC()), outer * inner * ((len + ch - 1) / ch), args);
    } else {
        long long o = outer, l = len, in = inner;
        float s = scale;
        void* args[] = {&pa, &py, &o, &l, &in, &s};
        cuda::launch(K("r_dim", REDUCE_SRC()), outer * inner, args);
    }
    attach(out, {a}, name, [a, dim, len, mean](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        TensorPtr g;
        if (dim == -1) {
            g = expand_to(reshape(G, {1}), a->shape());
        } else {
            Shape kept = a->shape();
            kept[dim] = 1;
            g = expand_to(reshape(G, kept), a->shape());
        }
        return {mean ? multiply(1.0f / (float)len, g) : g};
    });
    return out;
}

TensorPtr softmax(const TensorPtr& a0) {
    auto a = f32(a0, "softmax");
    if (!a->is_contiguous()) a = contiguous(a);
    if (a->dim() < 1) throw runtime_error("softmax: needs at least one dimension");
    long long cols = a->shape().back();
    const long long rows = cols ? a->numel() / cols : 0;
    auto out = empty_cuda(a->shape());
    const void* pa = base(a);
    void* py = out->storage()->data();
    void* args[] = {&pa, &py, &cols};
    if (rows) cuda::launch(K("r_softmax", REDUCE_SRC()), (unsigned)rows, 1, 1, 256, 1, 1, 0, args);
    attach(out, {a}, "softmax", [](const TensorPtr& out, const TensorPtr& G) -> vector<TensorPtr> {
        if (!grad_enabled()) return {softmax_backward(out, G)};
        auto s = sum(multiply(G, out), out->dim() - 1, true);
        return {multiply(out, subtract(G, s))};
    });
    return out;
}

void accumulate(void* grad, int64_t n, const TensorPtr& g0, bool assign) {
    auto g = on_device(g0, "accumulate");
    Index ix = index_for(g, g->shape());
    float* pg = static_cast<float*>(grad);
    const void* pa = base(g);
    int da = (int)g->dtype();
    long long nn = n;
    int as = assign ? 1 : 0;
    if (da == 0 && vec4(nn, ix, pa, pg)) {
        long long n4 = nn / 4;
        void* args[] = {&pg, &pa, &n4, &as};
        cuda::launch(K("k_accumulate4", CAST_SRC()), n4, args);
        return;
    }
    void* args[] = {&pg, &pa, &da, &nn, &ix, &as};
    cuda::launch(K("k_accumulate", CAST_SRC()), nn, args);
}

void fill(void* dst, int64_t n, float value) {
    float* p = static_cast<float*>(dst);
    long long nn = n;
    void* args[] = {&p, &nn, &value};
    cuda::launch(K("k_fill", CAST_SRC()), nn, args);
}

} // namespace cuda_ops

// ── indexing: extremes, argmax, gather / scatter, concatenation ──────────────

namespace cuda_common {

const string& INDEX_SRC() {
    static const string s = src(R"(
// is (v, i) better than (bv, bi)? ties go to the lower index, as on the CPU
__device__ bool better(float v, long long i, float bv, long long bi, int is_max) {
    if (i < 0) return false;
    if (bi < 0) return true;
    if (v == bv) return i < bi;
    return is_max ? v > bv : v < bv;
}
// arg max / min of n values; one block
extern "C" __global__ void i_extreme(const float* a, long long n, int is_max, long long* out) {
    __shared__ float sv[1024];
    __shared__ long long si[1024];
    float bv = 0.0f; long long bi = -1;
    for (long long i = threadIdx.x; i < n; i += blockDim.x)
        if (better(a[i], i, bv, bi, is_max)) { bv = a[i]; bi = i; }
    sv[threadIdx.x] = bv; si[threadIdx.x] = bi;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w && better(sv[threadIdx.x + w], si[threadIdx.x + w], sv[threadIdx.x], si[threadIdx.x], is_max)) {
            sv[threadIdx.x] = sv[threadIdx.x + w]; si[threadIdx.x] = si[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = si[0];
}
// index of the largest value in each row; one block per row; float32 out
extern "C" __global__ void i_argmax_rows(const float* a, long long cols, float* out) {
    __shared__ float sv[256];
    __shared__ long long si[256];
    const float* x = a + (long long)blockIdx.x * cols;
    float bv = 0.0f; long long bi = -1;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x)
        if (better(x[j], j, bv, bi, 1)) { bv = x[j]; bi = j; }
    sv[threadIdx.x] = bv; si[threadIdx.x] = bi;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w && better(sv[threadIdx.x + w], si[threadIdx.x + w], sv[threadIdx.x], si[threadIdx.x], 1)) {
            sv[threadIdx.x] = sv[threadIdx.x + w]; si[threadIdx.x] = si[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = (float)si[0];
}
// y[i] = x[idx[i]], any element size; a negative index reads as zero
// the element at a device-held index, and its gradient back into place
extern "C" __global__ void i_pick(const float* a, const long long* idx, float* y) { y[0] = a[idx[0]]; }
extern "C" __global__ void i_put(const float* g, const long long* idx, float* dx) { dx[idx[0]] = g[0]; }
extern "C" __global__ void i_gather(const unsigned char* x, const long long* idx, unsigned char* y,
                                    long long n, int es) {
    long long i = gid(); if (i >= n) return;
    long long j = idx[i];
    for (int b = 0; b < es; b++) y[i * es + b] = j < 0 ? 0 : x[j * es + b];
}
// y[idx[i]] += g[i]; an index can repeat, hence the atomic
extern "C" __global__ void i_scatter_add(const float* g, const long long* idx, float* y, long long n) {
    long long i = gid(); if (i >= n) return;
    long long j = idx[i];
    if (j >= 0) atomicAdd(&y[j], g[i]);
}
// one input's slab of a concatenation viewed as [outer, total, inner]
extern "C" __global__ void i_slab(const unsigned char* src, unsigned char* dst, long long outer,
                                  long long span, long long row, long long offset, int es) {
    long long i = gid(); if (i >= outer * span) return;
    long long o = i / span, j = i % span;
    if (es == 4) { ((float*)dst)[o * row + offset + j] = ((const float*)src)[i]; return; }
    for (int b = 0; b < es; b++) dst[(o * row + offset + j) * es + b] = src[i * es + b];
}
// Up to 8 inputs in one launch, forward and back: a thread per output
// element finds its input by the column offsets. One launch per input made a
// GPT step's QKV weight concatenation 24 launches.
struct Slabs { const unsigned char* p[8]; long long off[9]; int n; };
extern "C" __global__ void i_slabs(Slabs s, unsigned char* dst, long long outer, long long row, int es) {
    long long i = gid(); if (i >= outer * row) return;
    const long long o = i / row, q = i % row;
    int k = 0;
    while (k + 1 < s.n && q >= s.off[k + 1]) k++;
    const long long span = s.off[k + 1] - s.off[k], si = o * span + (q - s.off[k]);
    if (es == 4) { ((float*)dst)[i] = ((const float*)s.p[k])[si]; return; }
    for (int b = 0; b < es; b++) dst[i * es + b] = s.p[k][si * es + b];
}
// float32 gradient G back out to every part that wants one (p[k] null: skip)
extern "C" __global__ void i_unslabs(const float* G, Slabs s, long long outer, long long row) {
    long long i = gid(); if (i >= outer * row) return;
    const long long o = i / row, q = i % row;
    int k = 0;
    while (k + 1 < s.n && q >= s.off[k + 1]) k++;
    if (!s.p[k]) return;
    const long long span = s.off[k + 1] - s.off[k];
    ((float*)s.p[k])[o * span + (q - s.off[k])] = G[i];
}
// the reverse: one input's slab read back out of the concatenation
extern "C" __global__ void i_unslab(const unsigned char* src, unsigned char* dst, long long outer,
                                    long long span, long long row, long long offset, int es) {
    long long i = gid(); if (i >= outer * span) return;
    long long o = i / span, j = i % span;
    if (es == 4) { ((float*)dst)[i] = ((const float*)src)[o * row + offset + j]; return; }
    for (int b = 0; b < es; b++) dst[i * es + b] = src[(o * row + offset + j) * es + b];
}
)");
    return s;
}

TensorPtr upload_indices(const shared_ptr<const vector<int64_t>>& idx) {
    auto t = make_shared<Tensor>(Shape{(int64_t)idx->size()}, DType::Int64);
    if (!idx->empty()) memcpy(t->storage()->data(), idx->data(), idx->size() * sizeof(int64_t));
    return t->to(Device::CUDA);
}

TensorPtr contiguous_on_device(const TensorPtr& t) { return t->is_contiguous() ? t : contiguous(t); }

} // namespace cuda_common

namespace cuda_ops {

TensorPtr extreme(const char* name, const TensorPtr& a0, bool is_max) {
    auto a = contiguous_on_device(f32(a0, name));
    if (a->numel() == 0) throw runtime_error(string(name) + ": empty tensor");
    auto where_t = empty_cuda({1}, DType::Int64);
    const void* pa = base(a);
    void* pi = where_t->storage()->data();
    long long n = a->numel();
    int mx = is_max ? 1 : 0;
    void* args[] = {&pa, &n, &mx, &pi};
    cuda::launch(K("i_extreme", INDEX_SRC()), 1, 1, 1, 1024, 1, 1, 0, args);
    // the value and its gradient read the index on the device: max_op used
    // to copy it back to the CPU, which stalled the stream every call
    auto out = empty_cuda({1});
    {
        const void* pidx = pi;
        void* po = out->storage()->data();
        void* a2[] = {&pa, &pidx, &po};
        cuda::launch(K("i_pick", INDEX_SRC()), 1, a2);
    }
    attach(out, {a}, name, [a, where_t](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        if (!grad_enabled() && G->dtype() == F32) {
            auto g = contiguous_on_device(G);
            auto dx = make_shared<Tensor>(a->shape(), F32, Device::CUDA);           // zeroed
            const void* pg = base(g);
            const void* pidx = where_t->storage()->data();
            void* pd = dx->storage()->data();
            void* a3[] = {&pg, &pidx, &pd};
            cuda::launch(K("i_put", INDEX_SRC()), 1, a3);
            return {dx};
        }
        // recorded (higher-order) gradients: the op form, which needs the index on the host
        int64_t arg = 0;
        cuda::copy_to_host(&arg, where_t->storage()->data(), sizeof arg);
        auto idx = make_shared<const vector<int64_t>>(1, arg);
        return {scatter_add_flat(reshape(G, {1}), idx, a->shape())};
    });
    return out;
}

TensorPtr argmax(const TensorPtr& a0) {
    auto a = contiguous_on_device(f32(a0, "argmax"));
    if (a->dim() < 1) throw runtime_error("argmax: needs at least one dimension");
    long long cols = a->shape().back();
    const long long rows = cols ? a->numel() / cols : 0;
    Shape out_shape(a->shape().begin(), a->shape().end() - 1);
    if (out_shape.empty()) out_shape = {1};
    auto out = empty_cuda(out_shape);
    const void* pa = base(a);
    void* py = out->storage()->data();
    void* args[] = {&pa, &cols, &py};
    if (rows) cuda::launch(K("i_argmax_rows", INDEX_SRC()), (unsigned)rows, 1, 1, 256, 1, 1, 0, args);
    return out;
}

TensorPtr gather(const TensorPtr& x0, const shared_ptr<const vector<int64_t>>& idx, const Shape& out_shape) {
    auto x = contiguous_on_device(on_device(x0, "gather_flat"));
    auto out = empty_cuda(out_shape, x->dtype());
    if ((int64_t)idx->size() != out->numel())
        throw runtime_error("gather_flat: " + to_string(idx->size()) + " indices for " +
                            to_string(out->numel()) + " outputs");
    for (int64_t j : *idx)
        if (j >= x->numel()) throw runtime_error("gather_flat: index " + to_string(j) + " out of range");
    auto di = upload_indices(idx);
    const void* px = base(x);
    const void* pi = di->storage()->data();
    void* py = out->storage()->data();
    long long n = out->numel();
    int es = (int)dtype_size(x->dtype());
    void* args[] = {&px, &pi, &py, &n, &es};
    cuda::launch(K("i_gather", INDEX_SRC()), n, args);
    attach(out, {x}, "gather", [x, idx](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        return {scatter_add_flat(G, idx, x->shape())};
    });
    return out;
}

TensorPtr scatter_add(const TensorPtr& g0, const shared_ptr<const vector<int64_t>>& idx, const Shape& out_shape) {
    auto g = contiguous_on_device(f32(g0, "scatter_add_flat"));
    if ((int64_t)idx->size() != g->numel())
        throw runtime_error("scatter_add_flat: " + to_string(idx->size()) + " indices for " +
                            to_string(g->numel()) + " values");
    auto out = make_shared<Tensor>(out_shape, F32, Device::CUDA);      // zeroed
    for (int64_t j : *idx)
        if (j >= out->numel()) throw runtime_error("scatter_add_flat: index " + to_string(j) + " out of range");
    auto di = upload_indices(idx);
    const void* pg = base(g);
    const void* pi = di->storage()->data();
    void* py = out->storage()->data();
    long long n = g->numel();
    void* args[] = {&pg, &pi, &py, &n};
    cuda::launch(K("i_scatter_add", INDEX_SRC()), n, args);
    attach(out, {g}, "scatter_add", [g, idx](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        return {gather_flat(G, idx, g->shape())};
    });
    return out;
}

TensorPtr concat(const vector<TensorPtr>& inputs0, int axis) {
    if (inputs0.empty()) throw runtime_error("concat_op: no inputs");
    DType d = inputs0[0]->dtype();
    for (auto& t : inputs0) d = promote_types(d, t->dtype());
    vector<TensorPtr> c;
    for (auto& t : inputs0) c.push_back(contiguous_on_device(cast_op(on_device(t, "concat_op"), d)));
    const Shape shape0 = c[0]->shape();
    const int ndim = (int)shape0.size();
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) throw runtime_error("concat_op: axis out of range");
    int64_t total = 0;
    for (auto& t : c) {
        if (t->dim() != ndim) throw runtime_error("concat_op: inputs differ in rank");
        for (int k = 0; k < ndim; k++)
            if (k != axis && t->shape()[k] != shape0[k])
                throw runtime_error("concat_op: inputs differ on axis " + to_string(k));
        total += t->shape()[axis];
    }
    Shape out_shape = shape0;
    out_shape[axis] = total;
    auto out = empty_cuda(out_shape, d);
    long long outer = 1, inner = 1;
    for (int k = 0; k < axis; k++) outer *= shape0[k];
    for (int k = axis + 1; k < ndim; k++) inner *= shape0[k];
    long long offset = 0, row = total * inner;
    int es = (int)dtype_size(d);
    void* py = out->storage()->data();
    struct Slabs { const unsigned char* p[8]; long long off[9]; int n; };
    if (c.size() <= 8) {
        Slabs sl{};
        sl.n = (int)c.size();
        for (size_t k = 0; k < c.size(); k++) {
            sl.p[k] = static_cast<const unsigned char*>(base(c[k]));
            sl.off[k] = offset;
            offset += c[k]->shape()[axis] * inner;
        }
        sl.off[c.size()] = offset;
        long long o_ = outer;
        void* args[] = {&sl, &py, &o_, &row, &es};
        cuda::launch(K("i_slabs", INDEX_SRC()), outer * row, args);
    } else {
        for (auto& t : c) {
            long long span = t->shape()[axis] * inner;
            const void* ps = base(t);
            void* args[] = {&ps, &py, &outer, &span, &row, &offset, &es};
            cuda::launch(K("i_slab", INDEX_SRC()), outer * span, args);
            offset += span;
        }
    }
    attach(out, c, "concat_op", [c, outer, inner, total, axis](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        vector<TensorPtr> parts;
        int64_t off = 0;
        // a plain backward copies each slab out with one kernel; the gather
        // (an index list built and uploaded every call) only when recorded
        const bool plain = !grad_enabled() && G->dtype() == F32;
        auto Gc = plain ? contiguous_on_device(G) : G;
        if (plain && c.size() <= 8) {            // every part in one launch
            struct Slabs { const unsigned char* p[8]; long long off[9]; int n; };
            Slabs sl{};
            sl.n = (int)c.size();
            long long o2 = 0;
            for (size_t k = 0; k < c.size(); k++) {
                TensorPtr part = c[k]->requires_grad() ? empty_cuda(c[k]->shape()) : nullptr;
                sl.p[k] = part ? static_cast<const unsigned char*>(part->storage()->data()) : nullptr;
                sl.off[k] = o2;
                o2 += c[k]->shape()[axis] * inner;
                parts.push_back(part);
            }
            sl.off[c.size()] = o2;
            const void* pg = base(Gc);
            long long o_ = outer, row2 = total * inner;
            void* args[] = {&pg, &sl, &o_, &row2};
            cuda::launch(K("i_unslabs", INDEX_SRC()), outer * row2, args);
            return parts;
        }
        for (auto& t : c) {
            const int64_t len = t->shape()[axis];
            if (t->requires_grad() && plain) {
                auto part = empty_cuda(t->shape());
                long long o_ = outer, span = len * inner, row = total * inner, offset = off * inner;
                int es = 4;
                const void* ps = base(Gc);
                void* pd = part->storage()->data();
                void* args[] = {&ps, &pd, &o_, &span, &row, &offset, &es};
                cuda::launch(K("i_unslab", INDEX_SRC()), o_ * span, args);
                parts.push_back(part);
            } else if (t->requires_grad()) {
                auto idx = make_shared<vector<int64_t>>();
                idx->reserve((size_t)t->numel());
                for (int64_t o = 0; o < outer; o++)
                    for (int64_t j = 0; j < len; j++)
                        for (int64_t r = 0; r < inner; r++)
                            idx->push_back((o * total + off + j) * inner + r);
                parts.push_back(gather_flat(G, idx, t->shape()));
            } else {
                parts.push_back(nullptr);
            }
            off += len;
        }
        return parts;
    });
    return out;
}

// ── fused softmax and cross-entropy backward (see BACKWARD_SRC below) ────────

static const string& SOFTMAX_BACK_SRC() {
    static const string s = src(R"(
// dx = y * (g - sum(g * y)) over each row; one block per row
extern "C" __global__ void b_softmax(const float* y, const float* g, float* dx, long long cols) {
    __shared__ float s[256];
    const float* yr = y + (long long)blockIdx.x * cols;
    const float* gr = g + (long long)blockIdx.x * cols;
    float* dr = dx + (long long)blockIdx.x * cols;
    float t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) t += gr[j] * yr[j];
    s[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w]; __syncthreads(); }
    const float dot = s[0];
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) dr[j] = yr[j] * (gr[j] - dot);
}
// cross-entropy: dx = (softmax(x) - onehot(target)) * G / rows; one block per row
extern "C" __global__ void b_cross_entropy(const float* x, const long long* target, const float* G, float* dx,
                                           long long cols, long long rows) {
    __shared__ float s[256];
    const float* xr = x + (long long)blockIdx.x * cols;
    float* dr = dx + (long long)blockIdx.x * cols;
    float m = -3.402823466e38f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, xr[j]);
    s[threadIdx.x] = m;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + w]); __syncthreads(); }
    m = s[0];
    __syncthreads();
    float t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) t += expf(xr[j] - m);
    s[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w]; __syncthreads(); }
    const float inv = 1.0f / s[0], scale = G[0] / (float)rows;
    const long long tg = target[blockIdx.x];
    for (long long j = threadIdx.x; j < cols; j += blockDim.x)
        dr[j] = (expf(xr[j] - m) * inv - (j == tg ? 1.0f : 0.0f)) * scale;
}
)");
    return s;
}

TensorPtr softmax_backward(const TensorPtr& y0, const TensorPtr& G0) {
    auto y = contiguous_on_device(y0), G = contiguous_on_device(f32(G0, "softmax"));
    long long cols = y->shape().back();
    const long long rows = cols ? y->numel() / cols : 0;
    auto dx = empty_cuda(y->shape());
    const void* py = base(y);
    const void* pg = base(G);
    void* pd = dx->storage()->data();
    void* args[] = {&py, &pg, &pd, &cols};
    if (rows) cuda::launch(K("b_softmax", SOFTMAX_BACK_SRC()), (unsigned)rows, 1, 1, 256, 1, 1, 0, args);
    return dx;
}

TensorPtr cross_entropy_backward(const TensorPtr& x, const TensorPtr& targets, const TensorPtr& G0) {
    auto G = contiguous_on_device(f32(G0, "cross_entropy"));
    long long cols = x->shape().back();
    long long rows = cols ? x->numel() / cols : 0;
    auto dx = empty_cuda(x->shape());
    const void* px = base(x);
    const void* pt = targets->storage()->data();
    const void* pg = base(G);
    void* pd = dx->storage()->data();
    void* args[] = {&px, &pt, &pg, &pd, &cols, &rows};
    if (rows) cuda::launch(K("b_cross_entropy", SOFTMAX_BACK_SRC()), (unsigned)rows, 1, 1, 256, 1, 1, 0, args);
    return dx;
}

// ── cross-entropy over rows of logits ────────────────────────────────────────

static const string& LOSS_SRC() {
    static const string s = src(R"(
// loss[r] = log(sum_j exp(x[r, j])) - x[r, target[r]], stable; one block per row
extern "C" __global__ void l_cross_entropy(const float* x, const long long* target, float* loss, long long cols) {
    __shared__ float s[256];
    const float* row = x + (long long)blockIdx.x * cols;
    float m = -3.402823466e38f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, row[j]);
    s[threadIdx.x] = m;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + w]);
        __syncthreads();
    }
    m = s[0];
    __syncthreads();
    float t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) t += expf(row[j] - m);
    s[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w];
        __syncthreads();
    }
    // a target outside [0, cols) -- device targets are not checked on the host,
    // which would mean reading them back -- makes that row's loss NaN
    if (threadIdx.x == 0) {
        const long long tg = target[blockIdx.x];
        loss[blockIdx.x] = (tg < 0 || tg >= cols) ? __int_as_float(0x7fc00000) : logf(s[0]) + m - row[tg];
    }
}
)");
    return s;
}

// the loss with the targets already on the device as int64 [rows]
static TensorPtr cross_entropy_on(const TensorPtr& x, const TensorPtr& td) {
    long long cols = x->shape().back();
    const int64_t rows = cols ? x->numel() / cols : 0;
    auto per_row = empty_cuda({rows});
    const void* px = base(x);
    const void* pt = td->storage()->data();
    void* pl = per_row->storage()->data();
    void* args[] = {&px, &pt, &pl, &cols};
    if (rows) cuda::launch(K("l_cross_entropy", LOSS_SRC()), (unsigned)rows, 1, 1, 256, 1, 1, 0, args);
    TensorPtr out;
    {
        NoGrad off;
        out = ::mean(per_row, -1);
    }
    // d/dx = (softmax(x) - onehot(target)) / rows, written with ops so the
    // gradient is itself differentiable
    attach(out, {x}, "cross_entropy", [x, rows, cols, td](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        if (!grad_enabled()) return {cross_entropy_backward(x, td, G)};
        // the op form (recorded gradients) needs the targets on the host
        auto th = td->to(Device::CPU);
        auto pick = make_shared<vector<int64_t>>((size_t)rows);   // flat index of each target logit
        for (int64_t r = 0; r < rows; r++)
            (*pick)[(size_t)r] = r * cols + static_cast<const int64_t*>(th->storage()->data())[r];
        auto ones = make_shared<Tensor>(Shape{rows}, F32, Device::CUDA);
        fill(ones->storage()->data(), rows, 1.0f);
        auto onehot = scatter_add_flat(ones, pick, x->shape());
        auto g = divide(subtract(::softmax(x), onehot), (float)rows);
        return {multiply(reshape(G, {1}), g)};
    });
    return out;
}

TensorPtr cross_entropy(const TensorPtr& logits0, const vector<int>& targets) {
    auto x = contiguous_on_device(f32(logits0, "cross_entropy"));
    long long cols = x->shape().back();
    const int64_t rows = cols ? x->numel() / cols : 0;
    if ((int64_t)targets.size() != rows)
        throw runtime_error("cross_entropy: " + to_string(targets.size()) + " targets for " +
                            to_string(rows) + " positions");
    auto t = make_shared<Tensor>(Shape{rows}, DType::Int64);
    for (int64_t r = 0; r < rows; r++) {
        if (targets[r] < 0 || targets[r] >= cols)
            throw runtime_error("cross_entropy: target " + to_string(targets[r]) + " out of range");
        static_cast<int64_t*>(t->storage()->data())[r] = targets[r];
    }
    return cross_entropy_on(x, t->to(Device::CUDA));
}

TensorPtr cross_entropy_targets(const TensorPtr& logits0, const TensorPtr& targets0) {
    auto x = contiguous_on_device(f32(logits0, "cross_entropy"));
    long long cols = x->shape().back();
    const int64_t rows = cols ? x->numel() / cols : 0;
    if (targets0->numel() != rows)
        throw runtime_error("cross_entropy: " + to_string(targets0->numel()) + " targets for " +
                            to_string(rows) + " positions");
    if (targets0->dtype() != DType::Int64 && targets0->dtype() != DType::Int32)
        throw runtime_error(string("cross_entropy: targets must be int64 or int32, got ") +
                            dtype_name(targets0->dtype()));
    TensorPtr t = targets0->device() == Device::CUDA ? targets0 : targets0->to(Device::CUDA);
    if (t->dtype() != DType::Int64) t = cast(t, DType::Int64);
    return cross_entropy_on(x, contiguous_on_device(t));
}

// ── matmul on cuBLAS ─────────────────────────────────────────────────────────

// A matrix operand as cuBLAS reads it: contiguous, or the transpose view of
// contiguous memory (the last two dims swapped) read through the transpose
// flag instead of copied -- x @ W^T and attention's Q @ K^T. Anything else is
// made contiguous first.
struct Operand {
    TensorPtr t;
    bool trans;
    int ld;              // row length of the matrix as stored
};

static Operand operand(const TensorPtr& t) {
    const int nd = t->dim();
    const int64_t r = t->size(-2), c = t->size(-1);
    if (t->is_contiguous()) return {t, false, (int)c};
    const Shape& st = t->strides();
    bool tr = st[nd - 2] == 1 && st[nd - 1] == r;
    int64_t expect = r * c;
    for (int d = nd - 3; d >= 0 && tr; d--) {
        if (t->size(d) != 1 && st[d] != expect) tr = false;
        expect *= t->size(d);
    }
    if (tr) return {t, true, (int)r};
    return {contiguous(t), false, (int)c};
}

// The product alone, no gradient node. half: a and b are both of half dtype d
// and multiply through GemmEx, the result in float32 or (out_half) in d.
static TensorPtr matmul_raw(const TensorPtr& a, const TensorPtr& b, bool half, DType d, bool out_half) {
    Operand A = operand(a), B = operand(b);
    const size_t es = dtype_size(a->dtype());
    const DType od = out_half ? d : F32;
    const size_t os = dtype_size(od);
    auto gemm = [&](int m_, int n_, int k_, const void* pa, long long sa, const void* pb, long long sb,
                    void* pc, long long sc, int batch) {
        if (half)
            cuda::gemm_half(A.trans, B.trans, m_, n_, k_, d, pa, A.ld, sa, pb, B.ld, sb, pc, n_, sc, batch, out_half);
        else if (batch == 1)
            cuda::sgemm(A.trans, B.trans, m_, n_, k_, 1.0f, static_cast<const float*>(pa), A.ld,
                        static_cast<const float*>(pb), B.ld, 0.0f, static_cast<float*>(pc), n_);
        else
            cuda::sgemm_batched(A.trans, B.trans, m_, n_, k_, 1.0f, static_cast<const float*>(pa), A.ld, sa,
                                static_cast<const float*>(pb), B.ld, sb, 0.0f, static_cast<float*>(pc), n_, sc,
                                batch);
    };
    auto at = [&](const TensorPtr& t, int64_t elems) {
        return static_cast<const void*>(static_cast<const unsigned char*>(base(t)) + elems * es);
    };
    const int64_t m = a->size(-2), k = a->size(-1), n = b->size(-1);
    TensorPtr out;
    if (b->dim() == 2 && (!A.trans || a->dim() == 2)) {
        // every leading dimension of a folds into the rows: one gemm
        Shape out_shape = a->shape();
        out_shape.back() = n;
        out = make_shared<Tensor>(out_shape, od, Device::CUDA);
        if (k > 0)
            gemm((int)(a->numel() / k), (int)n, (int)k, base(A.t), 0, base(B.t), 0, out->storage()->data(), 0, 1);
    } else {
        if (b->dim() == 2) A = {contiguous(a), false, (int)k};    // a transposed N-D a cannot fold
        const Shape batch_a(a->shape().begin(), a->shape().end() - 2);
        const Shape batch_b(b->shape().begin(), b->shape().end() - 2);
        const Shape batch = broadcast_shapes(batch_a, batch_b);
        Shape out_shape = batch;
        out_shape.push_back(m);
        out_shape.push_back(n);
        out = make_shared<Tensor>(out_shape, od, Device::CUDA);
        int64_t nb = 1;
        for (int64_t v : batch) nb *= v;
        unsigned char* pc = static_cast<unsigned char*>(out->storage()->data());
        if (k > 0 && batch_a == batch && batch_b == batch) {
            gemm((int)m, (int)n, (int)k, base(A.t), m * k, base(B.t), k * n, pc, m * n, (int)nb);
        } else if (k > 0) {
            // broadcast batch dims: one product per output matrix
            vector<int64_t> ia((size_t)nb, 0), ib((size_t)nb, 0);
            broadcast_walk(batch, broadcast_strides(batch_a, batch), broadcast_strides(batch_b, batch), false,
                           [&](int64_t i, int64_t x, int64_t y) { ia[i] = x; ib[i] = y; });
            for (int64_t q = 0; q < nb; q++)
                gemm((int)m, (int)n, (int)k, at(A.t, ia[q] * m * k), 0, at(B.t, ib[q] * k * n), 0,
                     pc + q * m * n * os, 0, 1);
        }
    }
    return out;
}

TensorPtr matmul(const TensorPtr& a0, const TensorPtr& b0) {
    // Two float16 (or, on compute 8.0+, bfloat16) operands multiply as they
    // are, through GemmEx: float32 accumulation, tensor cores where present,
    // the half result written by the gemm itself. Anything else multiplies in
    // float32; a half result is rounded back.
    const DType d = result_type(a0, b0);
    const bool half_out = d == DType::Float16 || d == DType::BFloat16;
    const bool half_in = half_out && a0->dtype() == d && b0->dtype() == d &&
                         (d == DType::Float16 || cuda::has_bf16());
    auto a = half_in ? on_device(a0, "matmul") : f32(a0, "matmul");
    auto b = half_in ? on_device(b0, "matmul") : f32(b0, "matmul");
    auto out = matmul_raw(a, b, half_in, d, half_in);
    // the op form: any dtype, and what higher-order gradients differentiate
    auto graph = [a, b](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        TensorPtr gb;
        if (b->requires_grad()) {
            if (b->dim() == 2 && a->dim() > 2) {
                // a weight: every row of a contributes, so it is one product
                // over all of them -- not one per batch entry summed after
                const int64_t kk = a->size(-1), rows = a->numel() / kk;
                gb = ::matmul(transpose(reshape(a, {rows, kk})), reshape(G, {rows, G->size(-1)}));
            } else {
                gb = sum_to(::matmul(transpose(a), G), b->shape());
            }
        }
        return {a->requires_grad() ? sum_to(::matmul(G, transpose(b)), a->shape()) : nullptr, gb};
    };
    if (!half_in) {
        attach(out, {a, b}, "matmul", graph);
        return half_out ? cast_op(out, d) : out;
    }
    // half inputs: the backward multiplies in half too -- the incoming float32
    // gradient rounded to d once, both products through GemmEx, float32 out.
    // (Multiplying the float32 gradient by the half operands would promote
    // them back to float32: the backward of a mixed-precision step at full
    // float32 cost.)
    attach(out, {a, b}, "matmul", [graph, a, b, d](const TensorPtr& o, const TensorPtr& G0) -> vector<TensorPtr> {
        if (grad_enabled()) return graph(o, G0);
        auto G = G0->dtype() == d ? on_device(G0, "matmul") : cast(contiguous_on_device(G0), d);
        TensorPtr ga, gb;
        if (a->requires_grad()) ga = sum_to(matmul_raw(G, transpose(b), true, d, false), a->shape());
        if (b->requires_grad()) {
            if (b->dim() == 2 && a->dim() > 2) {
                const int64_t kk = a->size(-1), rows = a->numel() / kk;
                gb = matmul_raw(transpose(reshape(a, {rows, kk})), reshape(G, {rows, G->size(-1)}), true, d, false);
            } else {
                gb = sum_to(matmul_raw(transpose(a), G, true, d, false), b->shape());
            }
        }
        return {ga, gb};
    });
    return out;
}

// ── fused LayerNorm forward, dropout masks, optimizer updates ─────────────────

static const string& FUSED_SRC() {
    static const string s = src(R"(
// y = (x - mean) / sqrt(var + eps) * gamma + beta over rows of `cols`; one block per row
// y in the dtype ioy (float32, or half for the gemm it feeds under autocast)
extern "C" __global__ void f_layernorm(const float* x, const float* g, const float* b, void* y,
                                       long long cols, float eps, int ioy) {
    __shared__ float s[256];
    const float* row = x + (long long)blockIdx.x * cols;
    const long long yo = (long long)blockIdx.x * cols;
    float t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) t += row[j];
    s[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w]; __syncthreads(); }
    const float mean = s[0] / (float)cols;
    __syncthreads();
    t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) { float d = row[j] - mean; t += d * d; }
    s[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w]; __syncthreads(); }
    const float inv = rsqrtf(s[0] / (float)cols + eps);
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) stx(y, ioy, yo + j, (row[j] - mean) * inv * g[j] + b[j]);
}

// The same, one warp per row (8 rows a block) with the row held in registers
// (up to 32 values a lane: rows of up to 1024): one read of x, shuffle sums.
// A block per row spent its time in __syncthreads and re-read the row.
__device__ float lnw_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
extern "C" __global__ void f_layernorm_w(const float* x, const float* g, const float* b, void* y,
                                         int cols, float eps, long long rows, int ioy) {
    const long long r = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int lane = threadIdx.x & 31;
    const float* row = x + r * cols;
    float v[32];
    float t = 0.0f;
    #pragma unroll
    for (int k = 0; k < 32; k++) { const int j = lane + 32 * k; v[k] = j < cols ? row[j] : 0.0f; t += v[k]; }
    const float mean = lnw_sum(t) / (float)cols;
    t = 0.0f;
    #pragma unroll
    for (int k = 0; k < 32; k++) { const int j = lane + 32 * k; if (j < cols) { const float d = v[k] - mean; t += d * d; } }
    const float inv = rsqrtf(lnw_sum(t) / (float)cols + eps);
    const long long yo = r * cols;
    #pragma unroll
    for (int k = 0; k < 32; k++) { const int j = lane + 32 * k; if (j < cols) stx(y, ioy, yo + j, (v[k] - mean) * inv * g[j] + b[j]); }
}

// Philox-4x32-10 (Salmon et al. 2011, what cuRAND and PyTorch use): element
// i's uniform comes from counter i and the key, so the mask is the same
// whatever the launch shape, and a seed reproduces it exactly
__device__ unsigned int mulhi(unsigned int a, unsigned int b) { return (unsigned int)(((unsigned long long)a * b) >> 32); }
__device__ float philox_uniform(unsigned long long seed, unsigned long long i) {
    unsigned int c0 = (unsigned int)i, c1 = (unsigned int)(i >> 32), c2 = 0, c3 = 0;
    unsigned int k0 = (unsigned int)seed, k1 = (unsigned int)(seed >> 32);
    for (int r = 0; r < 10; r++) {
        unsigned int h0 = mulhi(0xD2511F53u, c0), l0 = 0xD2511F53u * c0;
        unsigned int h1 = mulhi(0xCD9E8D57u, c2), l1 = 0xCD9E8D57u * c2;
        unsigned int n0 = h1 ^ c1 ^ k0, n2 = h0 ^ c3 ^ k1;
        c0 = n0; c1 = l1; c2 = n2; c3 = l0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return (c0 >> 8) * (1.0f / 16777216.0f);      // 24 bits: [0, 1)
}
extern "C" __global__ void f_dropout_mask(float* mask, long long n, unsigned long long seed, float rate, float scale) {
    long long i = gid(); if (i >= n) return;
    mask[i] = philox_uniform(seed, (unsigned long long)i) < rate ? 0.0f : scale;
}

// optimizer updates, one thread per parameter element
extern "C" __global__ void o_sgd(float* w, const float* g, float* v, long long n, float lr, float mu) {
    long long i = gid(); if (i >= n) return;
    if (v) { v[i] = mu * v[i] + g[i]; w[i] -= lr * v[i]; }
    else w[i] -= lr * g[i];
}
extern "C" __global__ void o_adam(float* w, const float* g, float* m, float* v, long long n, float lr,
                                  float b1, float b2, float bc1, float bc2, float eps, float decay) {
    long long i = gid(); if (i >= n) return;
    float gi = g[i];
    m[i] = b1 * m[i] + (1.0f - b1) * gi;
    v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
    w[i] = w[i] * decay - lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + eps);
}
// Every parameter in one launch: a table of the tensors and a list of chunks
// of 8192 elements, a block per chunk. One launch per parameter tensor was
// ~50 small kernels a GPT step, most too small to fill the GPU.
struct MultiTensor { float* w; const float* g; float* m; float* v; long long n; };
// tdev: the step count on the device (capturable: a replayed step reads the
// count the previous replay left), or null to use bc1 / bc2 as given
extern "C" __global__ void o_adam_multi(const MultiTensor* t, const long long* chunk_start, const int* chunk_tensor,
                                        float lr, float b1, float b2, float bc1, float bc2, float eps, float decay,
                                        const float* tdev) {
    if (tdev) { bc1 = 1.0f - powf(b1, *tdev); bc2 = 1.0f - powf(b2, *tdev); }
    const MultiTensor T = t[chunk_tensor[blockIdx.x]];
    const long long s = chunk_start[blockIdx.x], e = min(T.n, s + 8192);
    for (long long i = s + threadIdx.x; i < e; i += blockDim.x) {
        const float gi = T.g[i];
        const float mi = b1 * T.m[i] + (1.0f - b1) * gi;
        const float vi = b2 * T.v[i] + (1.0f - b2) * gi * gi;
        T.m[i] = mi; T.v[i] = vi;
        T.w[i] = T.w[i] * decay - lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
    }
}
// Sum of squares of every gradient into one float on the device (zeroed
// first): a block per chunk, a tree in shared memory, one atomic a block.
extern "C" __global__ void o_sumsq_multi(const MultiTensor* t, const long long* chunk_start, const int* chunk_tensor,
                                         float* total) {
    __shared__ float sh[256];
    const MultiTensor T = t[chunk_tensor[blockIdx.x]];
    const long long s = chunk_start[blockIdx.x], e = min(T.n, s + 8192);
    float acc = 0.0f;
    for (long long i = s + threadIdx.x; i < e; i += blockDim.x) { const float v = T.g[i]; acc += v * v; }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) { if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(total, sh[0]);
}
// g *= max_norm / norm where the norm (sqrt of *total, read here on the
// device) exceeds max_norm: the CPU never waits for the norm to clip
extern "C" __global__ void o_clip_multi(const MultiTensor* t, const long long* chunk_start, const int* chunk_tensor,
                                        const float* total, float max_norm) {
    const float norm = sqrtf(*total);
    if (!(norm > max_norm)) return;
    const float c = max_norm / norm;
    const MultiTensor T = t[chunk_tensor[blockIdx.x]];
    float* g = (float*)T.g;
    const long long s = chunk_start[blockIdx.x], e = min(T.n, s + 8192);
    for (long long i = s + threadIdx.x; i < e; i += blockDim.x) g[i] *= c;
}
// the GradScaler's unscale over every gradient: g *= inv, or 0 and a flag
// where g is not finite
extern "C" __global__ void o_unscale_multi(const MultiTensor* t, const long long* chunk_start, const int* chunk_tensor,
                                           float inv, int* bad) {
    const MultiTensor T = t[chunk_tensor[blockIdx.x]];
    float* g = (float*)T.g;
    const long long s = chunk_start[blockIdx.x], e = min(T.n, s + 8192);
    for (long long i = s + threadIdx.x; i < e; i += blockDim.x) {
        const float v = g[i];
        if (isinf(v) || isnan(v)) { *bad = 1; g[i] = 0.0f; } else g[i] = v * inv;
    }
}
// one more step on the device-held step count
extern "C" __global__ void o_step_inc(float* t) { if (threadIdx.x == 0 && blockIdx.x == 0) t[0] += 1.0f; }
// SGD, the same way; m is the velocity (or null for none)
extern "C" __global__ void o_sgd_multi(const MultiTensor* t, const long long* chunk_start, const int* chunk_tensor,
                                       float lr, float mu) {
    const MultiTensor T = t[chunk_tensor[blockIdx.x]];
    const long long s = chunk_start[blockIdx.x], e = min(T.n, s + 8192);
    for (long long i = s + threadIdx.x; i < e; i += blockDim.x) {
        if (T.m) { const float vi = mu * T.m[i] + T.g[i]; T.m[i] = vi; T.w[i] -= lr * vi; }
        else T.w[i] -= lr * T.g[i];
    }
}
extern "C" __global__ void o_rmsprop(float* w, const float* g, float* v, long long n, float lr, float b, float eps) {
    long long i = gid(); if (i >= n) return;
    float gi = g[i];
    v[i] = b * v[i] + (1.0f - b) * gi * gi;
    w[i] -= lr * gi / (sqrtf(v[i]) + eps);
}
// sum of squares, one partial per block; and in-place scaling
extern "C" __global__ void o_sumsq(const float* a, float* part, long long n) {
    __shared__ float s[256];
    float acc = 0.0f;
    for (long long i = gid(); i < n; i += (long long)gridDim.x * blockDim.x) acc += a[i] * a[i];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w]; __syncthreads(); }
    if (threadIdx.x == 0) part[blockIdx.x] = s[0];
}
// g *= inv, or 0 and a flag where g is not finite (the GradScaler's unscale)
extern "C" __global__ void o_unscale(float* g, long long n, float inv, int* bad) {
    long long i = gid(); if (i >= n) return;
    float v = g[i];
    if (isinf(v) || isnan(v)) { *bad = 1; g[i] = 0.0f; } else g[i] = v * inv;
}
extern "C" __global__ void o_scale(float* a, long long n, float c) {
    long long i = gid(); if (i >= n) return;
    a[i] *= c;
}
)");
    return s;
}

TensorPtr layernorm(const TensorPtr& x0, const TensorPtr& g0, const TensorPtr& b0, float eps, DType out_dtype) {
    auto x = contiguous_on_device(f32(x0, "LayerNorm"));
    auto g = contiguous_on_device(f32(g0, "LayerNorm")), b = contiguous_on_device(f32(b0, "LayerNorm"));
    long long cols = x->shape().back();
    if (g->numel() != cols || b->numel() != cols)
        throw runtime_error("LayerNorm: gamma and beta must have " + to_string(cols) + " elements");
    const long long rows = cols ? x->numel() / cols : 0;
    auto out = empty_cuda(x->shape(), out_dtype);
    int ioy = out_dtype == DType::BFloat16 ? 3 : out_dtype == DType::Float16 ? 2 : 0;
    const void* px = base(x);
    const void* pg = base(g);
    const void* pb = base(b);
    void* py = out->storage()->data();
    if (rows && cols <= 1024) {
        int c = (int)cols;
        long long rr = rows;
        void* a[] = {&px, &pg, &pb, &py, &c, &eps, &rr, &ioy};
        cuda::launch(K("f_layernorm_w", FUSED_SRC()), (unsigned)((rows + 7) / 8), 1, 1, 256, 1, 1, 0, a);
        return out;
    }
    void* args[] = {&px, &pg, &pb, &py, &cols, &eps, &ioy};
    if (rows) cuda::launch(K("f_layernorm", FUSED_SRC()), (unsigned)rows, 1, 1, 256, 1, 1, 0, args);
    return out;
}

TensorPtr dropout_mask(const Shape& shape, uint64_t seed, float rate) {
    auto mask = empty_cuda(shape);
    void* pm = mask->storage()->data();
    long long n = mask->numel();
    unsigned long long s = seed;
    float r = rate, scale = 1.0f / (1.0f - rate);
    void* args[] = {&pm, &n, &s, &r, &scale};
    cuda::launch(K("f_dropout_mask", FUSED_SRC()), n, args);
    return mask;
}

void sgd_step(float* w, const float* g, float* v, int64_t n, float lr, float mu) {
    long long nn = n;
    void* args[] = {&w, &g, &v, &nn, &lr, &mu};
    cuda::launch(K("o_sgd", FUSED_SRC()), nn, args);
}

void adam_step(float* w, const float* g, float* m, float* v, int64_t n, float lr, float b1, float b2,
               float bc1, float bc2, float eps, float decay) {
    long long nn = n;
    void* args[] = {&w, &g, &m, &v, &nn, &lr, &b1, &b2, &bc1, &bc2, &eps, &decay};
    cuda::launch(K("o_adam", FUSED_SRC()), nn, args);
}

// the table and chunk list for a multi-tensor update, on the device: one copy
static TensorPtr multi_table(const vector<OptTensor>& ts, unsigned& chunks) {
    struct Row { float* w; const float* g; float* m; float* v; long long n; };
    static_assert(sizeof(Row) == 40, "MultiTensor layout");
    vector<long long> start;
    vector<int> owner;
    for (size_t i = 0; i < ts.size(); i++)
        for (long long s = 0; s < ts[i].n; s += 8192) { start.push_back(s); owner.push_back((int)i); }
    chunks = (unsigned)start.size();
    const size_t rows = ts.size() * sizeof(Row), starts = start.size() * 8, owners = owner.size() * 4;
    vector<unsigned char> host(rows + starts + owners);
    for (size_t i = 0; i < ts.size(); i++) {
        Row r{ts[i].w, ts[i].g, ts[i].m, ts[i].v, ts[i].n};
        memcpy(host.data() + i * sizeof(Row), &r, sizeof(Row));
    }
    if (starts) memcpy(host.data() + rows, start.data(), starts);
    if (owners) memcpy(host.data() + rows + starts, owner.data(), owners);
    auto dev = empty_cuda({(long long)((host.size() + 7) / 8)}, DType::Int64);
    cuda::copy_to_device(dev->storage()->data(), host.data(), host.size());
    return dev;
}

void adam_multi(const vector<OptTensor>& ts, float lr, float b1, float b2, float bc1, float bc2, float eps,
                float decay, float* step) {
    if (ts.empty()) return;
    if (step) {
        void* a0[] = {&step};
        cuda::launch(K("o_step_inc", FUSED_SRC()), 1, a0);
    }
    unsigned chunks = 0;
    auto table = multi_table(ts, chunks);
    if (!chunks) return;
    unsigned char* base_p = static_cast<unsigned char*>(table->storage()->data());
    const void* pt = base_p;
    const void* ps = base_p + ts.size() * 40;
    const void* po = base_p + ts.size() * 40 + (size_t)chunks * 8;
    const float* tdev = step;
    void* args[] = {&pt, &ps, &po, &lr, &b1, &b2, &bc1, &bc2, &eps, &decay, &tdev};
    cuda::launch(K("o_adam_multi", FUSED_SRC()), chunks, 1, 1, 256, 1, 1, 0, args);
}

float clip_grad_norm_multi(const vector<OptTensor>& ts, float max_norm, double* sum_sq, bool read) {
    auto total = make_shared<Tensor>(Shape{1}, F32, Device::CUDA);              // zeroed
    float* pt = static_cast<float*>(total->storage()->data());
    unsigned chunks = 0;
    TensorPtr table = ts.empty() ? nullptr : multi_table(ts, chunks);
    if (chunks) {
        unsigned char* bp = static_cast<unsigned char*>(table->storage()->data());
        const void* p1 = bp;
        const void* p2 = bp + ts.size() * 40;
        const void* p3 = bp + ts.size() * 40 + (size_t)chunks * 8;
        void* a1[] = {&p1, &p2, &p3, &pt};
        cuda::launch(K("o_sumsq_multi", FUSED_SRC()), chunks, 1, 1, 256, 1, 1, 0, a1);
        if (max_norm >= 0.0f) {
            void* a2[] = {&p1, &p2, &p3, &pt, &max_norm};
            cuda::launch(K("o_clip_multi", FUSED_SRC()), chunks, 1, 1, 256, 1, 1, 0, a2);
        }
    }
    if (!read && !sum_sq) return std::numeric_limits<float>::quiet_NaN();   // no wait for the GPU
    float h = 0.0f;
    cuda::copy_to_host(&h, pt, sizeof h);        // the one read: the norm the caller returns
    if (sum_sq) *sum_sq = h;
    return std::sqrt(h);
}

bool unscale_multi(const vector<OptTensor>& ts, float inv) {
    if (ts.empty()) return true;
    unsigned chunks = 0;
    auto table = multi_table(ts, chunks);
    if (!chunks) return true;
    auto flag = make_shared<Tensor>(Shape{1}, DType::Int32, Device::CUDA);      // zeroed
    void* pf = flag->storage()->data();
    unsigned char* bp = static_cast<unsigned char*>(table->storage()->data());
    const void* p1 = bp;
    const void* p2 = bp + ts.size() * 40;
    const void* p3 = bp + ts.size() * 40 + (size_t)chunks * 8;
    void* args[] = {&p1, &p2, &p3, &inv, &pf};
    cuda::launch(K("o_unscale_multi", FUSED_SRC()), chunks, 1, 1, 256, 1, 1, 0, args);
    int bad = 0;
    cuda::copy_to_host(&bad, pf, sizeof bad);   // one read for every parameter
    return bad == 0;
}

void sgd_multi(const vector<OptTensor>& ts, float lr, float mu) {
    if (ts.empty()) return;
    unsigned chunks = 0;
    auto table = multi_table(ts, chunks);
    if (!chunks) return;
    unsigned char* base_p = static_cast<unsigned char*>(table->storage()->data());
    const void* pt = base_p;
    const void* ps = base_p + ts.size() * 40;
    const void* po = base_p + ts.size() * 40 + (size_t)chunks * 8;
    void* args[] = {&pt, &ps, &po, &lr, &mu};
    cuda::launch(K("o_sgd_multi", FUSED_SRC()), chunks, 1, 1, 256, 1, 1, 0, args);
}

void rmsprop_step(float* w, const float* g, float* v, int64_t n, float lr, float b, float eps) {
    long long nn = n;
    void* args[] = {&w, &g, &v, &nn, &lr, &b, &eps};
    cuda::launch(K("o_rmsprop", FUSED_SRC()), nn, args);
}

double sum_squares(const float* a, int64_t n) {
    if (n <= 0) return 0.0;
    const long long blocks = min<long long>((n + 255) / 256, 1024);
    auto part = empty_cuda({blocks});
    void* pp = part->storage()->data();
    long long nn = n;
    void* args[] = {&a, &pp, &nn};
    cuda::launch(K("o_sumsq", FUSED_SRC()), (unsigned)blocks, 1, 1, 256, 1, 1, 0, args);
    vector<float> h((size_t)blocks);
    cuda::copy_to_host(h.data(), pp, h.size() * sizeof(float));
    double total = 0.0;
    for (float v : h) total += v;
    return total;
}

bool unscale_inplace(float* g, int64_t n, float inv) {
    if (n <= 0) return true;
    auto flag = make_shared<Tensor>(Shape{1}, DType::Int32, Device::CUDA);      // zeroed
    void* pf = flag->storage()->data();
    long long nn = n;
    void* args[] = {&g, &nn, &inv, &pf};
    cuda::launch(K("o_unscale", FUSED_SRC()), nn, args);
    int bad = 0;
    cuda::copy_to_host(&bad, pf, sizeof bad);
    return bad == 0;
}

void scale_inplace(float* a, int64_t n, float c) {
    long long nn = n;
    void* args[] = {&a, &nn, &c};
    cuda::launch(K("o_scale", FUSED_SRC()), nn, args);
}

} // namespace cuda_ops

// ── fused element-wise functions (p.fuse) ─────────────────────────────────────
//
// An expression over inputs x0..x{k-1} (CUDA C, written by p.fuse's tracer)
// becomes one kernel: every input read through its broadcast index, the
// result written once. Its gradient is a second generated kernel: the
// upstream gradient g and the inputs in, every input's gradient (at the
// output's shape) out, reduced to each input's shape after. Sources are
// compiled once per expression and kept.

namespace {
constexpr int FUSE_MAX = 8;
struct FuseIn {
    const float* p[FUSE_MAX];
    Index ix[FUSE_MAX];
};
struct FuseOut {
    float* p[FUSE_MAX];
};
const char* const FUSE_HELPERS = R"(
#define FUSE_MAX 8
struct FuseIn { const float* p[FUSE_MAX]; Index ix[FUSE_MAX]; };
struct FuseOut { float* p[FUSE_MAX]; };
__device__ __forceinline__ float f_sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float f_gelu(float v) { return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))); }
__device__ __forceinline__ float f_gelu_d(float v) {
    const float k = 0.7978845608f, c = 0.044715f, t = tanhf(k * (v + c * v * v * v));
    return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * k * (1.0f + 3.0f * c * v * v);
}
)";
// fwd: statements defining `out` from x0..x{k-1}; bwd: statements defining
// d{j} for every j in mask, from the same and the upstream gradient g
const string& fuse_source(const string& fwd, const string& bwd, const vector<int>& mask, int k) {
    static std::mutex mu;
    static auto& cache = *new std::unordered_map<string, string>();     // Kernel keeps c_str()
    string key = fwd + "\x1f" + bwd + "\x1f" + to_string(k);
    for (int m : mask) key += m ? '1' : '0';
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    string loads;
    for (int j = 0; j < k; j++)
        loads += "    const float x" + to_string(j) + " = in.p[" + to_string(j) + "][at(in.ix[" + to_string(j) + "], i)];\n";
    string body = string(FUSE_HELPERS) +
        "extern \"C\" __global__ void fz_fwd(FuseIn in, float* y, long long n) {\n"
        "    const long long i = gid(); if (i >= n) return;\n" + loads + fwd + "\n    y[i] = out;\n}\n"
        "extern \"C\" __global__ void fz_bwd(FuseIn in, const float* G, FuseOut o, long long n) {\n"
        "    const long long i = gid(); if (i >= n) return;\n" + loads +
        "    const float g = G[i];\n" + bwd + "\n";
    for (int j = 0; j < k; j++)
        if (mask[j])                           // null: that input needs no gradient
            body += "    if (o.p[" + to_string(j) + "]) o.p[" + to_string(j) + "][i] = d" + to_string(j) + ";\n";
    body += "}\n";
    return cache.emplace(key, src(body.c_str())).first->second;
}
} // namespace

namespace cuda_ops {

TensorPtr fused_map(const string& fwd, const string& bwd, const vector<int>& mask, const vector<TensorPtr>& inputs0,
                    FusedHigher higher) {
    const int k = (int)inputs0.size();
    if (k < 1 || k > FUSE_MAX) throw runtime_error("fuse: 1 to 8 tensor inputs");
    if ((int)mask.size() != k) throw runtime_error("fuse: one gradient flag per input");
    vector<TensorPtr> in;
    Shape shape;
    for (int j = 0; j < k; j++) {
        in.push_back(f32(inputs0[j], "fuse"));
        shape = j ? broadcast_shapes(shape, in[j]->shape()) : in[j]->shape();
    }
    const string& source = fuse_source(fwd, bwd, mask, k);
    auto out = empty_cuda(shape);
    FuseIn fi{};
    for (int j = 0; j < k; j++) {
        fi.p[j] = static_cast<const float*>(base(in[j]));
        fi.ix[j] = index_for(in[j], shape);
    }
    float* py = static_cast<float*>(out->storage()->data());
    long long n = out->numel();
    void* a[] = {&fi, &py, &n};
    if (n) cuda::launch(K("fz_fwd", source), n, a);
    const string* srcp = &source;                  // in the never-destroyed cache
    attach(out, in, "fuse", [in, mask, shape, n, srcp, higher](const TensorPtr&, const TensorPtr& G0) -> vector<TensorPtr> {
        if (grad_enabled()) {
            // the gradient is itself recorded: each input's g * d/dx_j as a fused
            // op of its own (built by the caller from the expression tree), so it
            // differentiates again
            if (!higher) throw runtime_error("fuse: no higher-order gradient for this fused op");
            auto r = higher(in, G0);
            for (size_t j = 0; j < r.size() && j < in.size(); j++)
                if (r[j] && r[j]->shape() != in[j]->shape()) r[j] = sum_to(r[j], in[j]->shape());
            return r;
        }
        auto G = contiguous_on_device(f32(G0, "fuse"));
        const int kk = (int)in.size();
        FuseIn fi{};
        FuseOut fo{};
        vector<TensorPtr> full(kk);
        for (int j = 0; j < kk; j++) {
            fi.p[j] = static_cast<const float*>(base(in[j]));
            fi.ix[j] = index_for(in[j], shape);
            if (in[j]->requires_grad() && mask[j]) {
                full[j] = empty_cuda(shape);
                fo.p[j] = static_cast<float*>(full[j]->storage()->data());
            }
        }
        const float* pg = static_cast<const float*>(base(G));
        long long nn = n;
        void* a[] = {&fi, &pg, &fo, &nn};
        if (nn) cuda::launch(K("fz_bwd", *srcp), nn, a);
        vector<TensorPtr> r(kk);
        for (int j = 0; j < kk; j++)
            if (full[j]) r[j] = full[j]->shape() == in[j]->shape() ? full[j] : sum_to(full[j], in[j]->shape());
        return r;
    });
    return out;
}

} // namespace cuda_ops
