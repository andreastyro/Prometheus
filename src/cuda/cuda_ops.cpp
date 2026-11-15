// Ops on GPU tensors: kernels compiled at runtime by NVRTC. See ml/cuda.hpp.
//
// Every GPU op computes its forward with a kernel and attaches a
// backward_graph -- its gradient written with ops, which run on the GPU too.
// The autograd engine runs that graph for any node on the GPU, and the
// gradients accumulate on the device. There are no raw backward kernels to
// keep in step with the forward, and the gradients are differentiable again.
#include "../ops_internal.hpp"
#include "ml/cuda.hpp"
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

using namespace std;

namespace {

constexpr int MAX_DIMS = 8;
constexpr DType F32 = DType::Float32;

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
// element i of the output, read through a strided / broadcast operand
__device__ long long at(const Index& ix, long long i) {
    if (ix.nd < 0) return i;                     // contiguous, same shape
    long long r = i, o = 0;
    for (int d = ix.nd - 1; d >= 0; d--) { long long q = r % ix.shape[d]; r /= ix.shape[d]; o += q * ix.st[d]; }
    return o;
}
__device__ long long gid() { return (long long)blockIdx.x * blockDim.x + threadIdx.x; }
)";

struct Index {
    long long shape[MAX_DIMS];
    long long st[MAX_DIMS];
    int nd;
};

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
#define UNARY(NAME, EXPR) extern "C" __global__ void NAME(const float* a, float* y, long long n, Index ix, float p0, float p1) { \
    long long i = gid(); if (i >= n) return; float x = a[at(ix, i)]; y[i] = EXPR; }
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
    long long i = gid(); if (i >= n) return; float x = a[at(ia, i)], y = b[at(ib, i)]; c[i] = EXPR; }
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
extern "C" __global__ void k_accumulate(float* g, const void* a, int da, long long n, Index ix) {
    long long i = gid(); if (i >= n) return;
    g[i] += (float)ld(a, da, at(ix, i));
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

TensorPtr empty_cuda(const Shape& shape, DType d = F32) {
    return make_shared<Tensor>(shape, Tensor::Uninitialized{}, d, Device::CUDA);
}

const void* base(const TensorPtr& t) {
    return static_cast<const unsigned char*>(t->storage()->data()) + t->offset() * dtype_size(t->dtype());
}

void attach(const TensorPtr& out, vector<TensorPtr> inputs, const char* name,
            function<vector<TensorPtr>(const TensorPtr&, const TensorPtr&)> graph) {
    bool need = false;
    for (auto& t : inputs) need = need || (t && t->requires_grad());
    if (!need) return;
    auto node = make_node(out, std::move(inputs));
    node->name = name;
    node->backward_graph = std::move(graph);
}

} // namespace

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
    void* args[] = {&pa, &da, &py, &dy, &n, &ix};
    cuda::launch(K("k_cast", CAST_SRC()), n, args);
    return out;
}

TensorPtr copy(const TensorPtr& a) { return cast(a, a->dtype()); }

// operands for a float32 kernel: on the device, as float32
// through cast_op, so a half-precision input's gradient flows back through
// the conversion
static TensorPtr f32(const TensorPtr& t, const char* op) {
    auto d = on_device(t, op);
    return d->dtype() == F32 ? d : cast_op(d, F32);
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
    void* args[] = {&pa, &py, &n, &ix, &p0, &p1};
    cuda::launch(K(kernel, UNARY_SRC()), n, args);
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
    void* args[] = {&pa, &pb, &pc, &n, &ia, &ib};
    cuda::launch(K(names[op], BINARY_SRC()), n, args);
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

void accumulate(void* grad, int64_t n, const TensorPtr& g0) {
    auto g = on_device(g0, "accumulate");
    Index ix = index_for(g, g->shape());
    float* pg = static_cast<float*>(grad);
    const void* pa = base(g);
    int da = (int)g->dtype();
    long long nn = n;
    void* args[] = {&pg, &pa, &da, &nn, &ix};
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

namespace {

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

} // namespace

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
    int64_t arg = 0;
    cuda::copy_to_host(&arg, pi, sizeof arg);
    auto idx = make_shared<const vector<int64_t>>(1, arg);
    TensorPtr out;
    {
        NoGrad off;
        out = gather_flat(a, idx, Shape{1});
    }
    attach(out, {a}, name, [a, idx](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
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
    for (auto& t : c) {
        long long span = t->shape()[axis] * inner;
        const void* ps = base(t);
        void* args[] = {&ps, &py, &outer, &span, &row, &offset, &es};
        cuda::launch(K("i_slab", INDEX_SRC()), outer * span, args);
        offset += span;
    }
    attach(out, c, "concat_op", [c, outer, inner, total, axis](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        vector<TensorPtr> parts;
        int64_t off = 0;
        // a plain backward copies each slab out with one kernel; the gather
        // (an index list built and uploaded every call) only when recorded
        const bool plain = !grad_enabled() && G->dtype() == F32;
        auto Gc = plain ? contiguous_on_device(G) : G;
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
    if (threadIdx.x == 0) loss[blockIdx.x] = logf(s[0]) + m - row[target[blockIdx.x]];
}
)");
    return s;
}

TensorPtr cross_entropy(const TensorPtr& logits0, const vector<int>& targets) {
    auto x = contiguous_on_device(f32(logits0, "cross_entropy"));
    long long cols = x->shape().back();
    const int64_t rows = cols ? x->numel() / cols : 0;
    if ((int64_t)targets.size() != rows)
        throw runtime_error("cross_entropy: " + to_string(targets.size()) + " targets for " +
                            to_string(rows) + " positions");
    auto t = make_shared<Tensor>(Shape{rows}, DType::Int64);
    auto pick = make_shared<vector<int64_t>>((size_t)rows);   // flat index of each target logit
    for (int64_t r = 0; r < rows; r++) {
        if (targets[r] < 0 || targets[r] >= cols)
            throw runtime_error("cross_entropy: target " + to_string(targets[r]) + " out of range");
        static_cast<int64_t*>(t->storage()->data())[r] = targets[r];
        (*pick)[(size_t)r] = r * cols + targets[r];
    }
    auto td = t->to(Device::CUDA);
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
    attach(out, {x}, "cross_entropy", [x, pick, rows, td](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        if (!grad_enabled()) return {cross_entropy_backward(x, td, G)};
        auto ones = make_shared<Tensor>(Shape{rows}, F32, Device::CUDA);
        fill(ones->storage()->data(), rows, 1.0f);
        auto onehot = scatter_add_flat(ones, pick, x->shape());
        auto g = divide(subtract(::softmax(x), onehot), (float)rows);
        return {multiply(reshape(G, {1}), g)};
    });
    return out;
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

TensorPtr matmul(const TensorPtr& a0, const TensorPtr& b0) {
    // Two float16 (or, on compute 8.0+, bfloat16) operands multiply as they
    // are, through GemmEx: float32 accumulation, tensor cores where present.
    // Anything else multiplies in float32; a half result is rounded back.
    const DType d = result_type(a0, b0);
    const bool half_out = d == DType::Float16 || d == DType::BFloat16;
    const bool half_in = half_out && a0->dtype() == d && b0->dtype() == d &&
                         (d == DType::Float16 || cuda::has_bf16());
    auto a = half_in ? on_device(a0, "matmul") : f32(a0, "matmul");
    auto b = half_in ? on_device(b0, "matmul") : f32(b0, "matmul");
    Operand A = operand(a), B = operand(b);
    const size_t es = dtype_size(a->dtype());
    auto gemm = [&](int m_, int n_, int k_, const void* pa, long long sa, const void* pb, long long sb,
                    float* pc, long long sc, int batch) {
        if (half_in)
            cuda::gemm_half(A.trans, B.trans, m_, n_, k_, d, pa, A.ld, sa, pb, B.ld, sb, pc, n_, sc, batch);
        else if (batch == 1)
            cuda::sgemm(A.trans, B.trans, m_, n_, k_, 1.0f, static_cast<const float*>(pa), A.ld,
                        static_cast<const float*>(pb), B.ld, 0.0f, pc, n_);
        else
            cuda::sgemm_batched(A.trans, B.trans, m_, n_, k_, 1.0f, static_cast<const float*>(pa), A.ld, sa,
                                static_cast<const float*>(pb), B.ld, sb, 0.0f, pc, n_, sc, batch);
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
        out = make_shared<Tensor>(out_shape, F32, Device::CUDA);
        if (k > 0)
            gemm((int)(a->numel() / k), (int)n, (int)k, base(A.t), 0, base(B.t), 0,
                 static_cast<float*>(out->storage()->data()), 0, 1);
    } else {
        if (b->dim() == 2) A = {contiguous(a), false, (int)k};    // a transposed N-D a cannot fold
        const Shape batch_a(a->shape().begin(), a->shape().end() - 2);
        const Shape batch_b(b->shape().begin(), b->shape().end() - 2);
        const Shape batch = broadcast_shapes(batch_a, batch_b);
        Shape out_shape = batch;
        out_shape.push_back(m);
        out_shape.push_back(n);
        out = make_shared<Tensor>(out_shape, F32, Device::CUDA);
        int64_t nb = 1;
        for (int64_t v : batch) nb *= v;
        float* pc = static_cast<float*>(out->storage()->data());
        if (k > 0 && batch_a == batch && batch_b == batch) {
            gemm((int)m, (int)n, (int)k, base(A.t), m * k, base(B.t), k * n, pc, m * n, (int)nb);
        } else if (k > 0) {
            // broadcast batch dims: one product per output matrix
            vector<int64_t> ia((size_t)nb, 0), ib((size_t)nb, 0);
            broadcast_walk(batch, broadcast_strides(batch_a, batch), broadcast_strides(batch_b, batch), false,
                           [&](int64_t i, int64_t x, int64_t y) { ia[i] = x; ib[i] = y; });
            for (int64_t q = 0; q < nb; q++)
                gemm((int)m, (int)n, (int)k, at(A.t, ia[q] * m * k), 0, at(B.t, ib[q] * k * n), 0, pc + q * m * n, 0, 1);
        }
    }
    attach(out, {a, b}, "matmul", [a, b](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
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
    });
    return half_out ? cast_op(out, d) : out;
}

// ── fused LayerNorm forward, dropout masks, optimizer updates ─────────────────

static const string& FUSED_SRC() {
    static const string s = src(R"(
// y = (x - mean) / sqrt(var + eps) * gamma + beta over rows of `cols`; one block per row
extern "C" __global__ void f_layernorm(const float* x, const float* g, const float* b, float* y,
                                       long long cols, float eps) {
    __shared__ float s[256];
    const float* row = x + (long long)blockIdx.x * cols;
    float* out = y + (long long)blockIdx.x * cols;
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
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) out[j] = (row[j] - mean) * inv * g[j] + b[j];
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
extern "C" __global__ void o_adam_multi(const MultiTensor* t, const long long* chunk_start, const int* chunk_tensor,
                                        float lr, float b1, float b2, float bc1, float bc2, float eps, float decay) {
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

TensorPtr layernorm(const TensorPtr& x0, const TensorPtr& g0, const TensorPtr& b0, float eps) {
    auto x = contiguous_on_device(f32(x0, "LayerNorm"));
    auto g = contiguous_on_device(f32(g0, "LayerNorm")), b = contiguous_on_device(f32(b0, "LayerNorm"));
    long long cols = x->shape().back();
    if (g->numel() != cols || b->numel() != cols)
        throw runtime_error("LayerNorm: gamma and beta must have " + to_string(cols) + " elements");
    const long long rows = cols ? x->numel() / cols : 0;
    auto out = empty_cuda(x->shape());
    const void* px = base(x);
    const void* pg = base(g);
    const void* pb = base(b);
    void* py = out->storage()->data();
    void* args[] = {&px, &pg, &pb, &py, &cols, &eps};
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
                float decay) {
    if (ts.empty()) return;
    unsigned chunks = 0;
    auto table = multi_table(ts, chunks);
    if (!chunks) return;
    unsigned char* base_p = static_cast<unsigned char*>(table->storage()->data());
    const void* pt = base_p;
    const void* ps = base_p + ts.size() * 40;
    const void* po = base_p + ts.size() * 40 + (size_t)chunks * 8;
    void* args[] = {&pt, &ps, &po, &lr, &b1, &b2, &bc1, &bc2, &eps, &decay};
    cuda::launch(K("o_adam_multi", FUSED_SRC()), chunks, 1, 1, 256, 1, 1, 0, args);
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

// ── fused backward kernels ───────────────────────────────────────────────────
//
// A GPU op's backward_graph builds its gradient from ops, so it can be
// differentiated again -- but for LayerNorm, GELU and the embedding that is a
// dozen or more small kernels per call, most of the backward's time. When the
// gradient is not itself being recorded (a plain .backward(), or grad()
// without create_graph -- grad_enabled() is false), these fused kernels give
// the same numbers in one or two launches; with create_graph the op form runs.

static const string& BACKWARD_SRC() {
    static const string s = src(R"(
// LayerNorm backward, one warp per row (8 rows to a block), sums by shuffles:
// dx, and the row's mean and 1/std for the gamma / beta column sums.
//   dx = inv * (g*dy - mean(g*dy) - xhat * mean(g*dy*xhat))
// A block per row spent most of its time in __syncthreads on rows of a few
// hundred: 2x PyTorch's time on a GTX 970.
__device__ float ln_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
extern "C" __global__ void b_layernorm_rows(const float* x, const float* g, const float* dy, float* dx,
                                            float* mean_out, float* inv_out, long long cols, float eps,
                                            long long rows) {
    const long long r = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int lane = threadIdx.x & 31;
    const float* xr = x + r * cols;
    const float* dyr = dy + r * cols;
    float* dxr = dx + r * cols;
    float t = 0.0f;
    for (long long j = lane; j < cols; j += 32) t += xr[j];
    const float mean = ln_warp_sum(t) / (float)cols;
    t = 0.0f;
    for (long long j = lane; j < cols; j += 32) { const float d = xr[j] - mean; t += d * d; }
    const float inv = rsqrtf(ln_warp_sum(t) / (float)cols + eps);
    float a = 0.0f, b = 0.0f;
    for (long long j = lane; j < cols; j += 32) {
        const float gd = g[j] * dyr[j];
        a += gd;
        b += gd * (xr[j] - mean) * inv;
    }
    const float m1 = ln_warp_sum(a) / (float)cols, m2 = ln_warp_sum(b) / (float)cols;
    for (long long j = lane; j < cols; j += 32) {
        const float xh = (xr[j] - mean) * inv;
        dxr[j] = inv * (g[j] * dyr[j] - m1 - xh * m2);
    }
    if (lane == 0) { mean_out[r] = mean; inv_out[r] = inv; }
}
// gamma / beta gradients: a thread per (chunk of 32 rows, column), added
// atomically -- one thread per column left most of the GPU idle
extern "C" __global__ void b_layernorm_cols(const float* x, const float* dy, const float* mean, const float* inv,
                                            float* dg, float* db, long long rows, long long cols) {
    const long long chunks = (rows + 31) / 32;
    long long i = gid(); if (i >= cols * chunks) return;
    const long long j = i % cols, r0 = (i / cols) * 32, r1 = min(rows, r0 + 32);
    float ag = 0.0f, ab = 0.0f;
    for (long long r = r0; r < r1; r++) {
        const float d = dy[r * cols + j];
        ag += d * (x[r * cols + j] - mean[r]) * inv[r];
        ab += d;
    }
    atomicAdd(&dg[j], ag);
    atomicAdd(&db[j], ab);
}
// GELU (tanh form): dx = dy * d/dx [0.5 x (1 + tanh(k (x + c x^3)))]
extern "C" __global__ void b_gelu(const float* x, const float* dy, float* dx, long long n) {
    long long i = gid(); if (i >= n) return;
    const float k = 0.7978845608f, c = 0.044715f, v = x[i];
    const float t = tanhf(k * (v + c * v * v * v));
    dx[i] = dy[i] * (0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * k * (1.0f + 3.0f * c * v * v));
}
// embedding: rows of the table by id, and the scatter-add back (ids repeat).
// ids come from a device tensor (dtype code dt) or, with mod > 0, are the
// positions r % mod. Checking range on the host would mean reading the ids
// back every step; an id outside [0, vocab) reads as a zero row instead.
__device__ long long emb_id(const void* ids, int dt, long long r, long long mod) {
    if (mod > 0) return r % mod;
    return dt == 5 ? (long long)((const int*)ids)[r] : ((const long long*)ids)[r];
}
extern "C" __global__ void b_embed_fwd(const float* w, const void* ids, int dt, long long mod, float* out,
                                       long long rows, long long e, long long vocab) {
    long long i = gid(); if (i >= rows * e) return;
    const long long id = emb_id(ids, dt, i / e, mod);
    out[i] = (id < 0 || id >= vocab) ? 0.0f : w[id * e + i % e];
}
extern "C" __global__ void b_embed_bwd(const float* g, const void* ids, int dt, long long mod, float* dw,
                                       long long rows, long long e, long long vocab) {
    long long i = gid(); if (i >= rows * e) return;
    const long long id = emb_id(ids, dt, i / e, mod);
    if (id >= 0 && id < vocab) atomicAdd(&dw[id * e + i % e], g[i]);
}
)");
    return s;
}

vector<TensorPtr> layernorm_backward(const TensorPtr& x0, const TensorPtr& g0, const TensorPtr& G0, float eps) {
    auto x = contiguous_on_device(f32(x0, "LayerNorm")), g = contiguous_on_device(f32(g0, "LayerNorm"));
    auto G = contiguous_on_device(f32(G0, "LayerNorm"));
    long long cols = x->shape().back();
    long long rows = cols ? x->numel() / cols : 0;
    auto dx = empty_cuda(x->shape());
    auto dg = make_shared<Tensor>(Shape{cols}, F32, Device::CUDA), db = make_shared<Tensor>(Shape{cols}, F32, Device::CUDA);
    auto mean = empty_cuda({max(rows, 1LL)}), inv = empty_cuda({max(rows, 1LL)});
    const void* px = base(x);
    const void* pg = base(g);
    const void* pd = base(G);
    void* pdx = dx->storage()->data();
    void* pm = mean->storage()->data();
    void* pi = inv->storage()->data();
    void* pdg = dg->storage()->data();
    void* pdb = db->storage()->data();
    void* a1[] = {&px, &pg, &pd, &pdx, &pm, &pi, &cols, &eps, &rows};
    if (rows) cuda::launch(K("b_layernorm_rows", BACKWARD_SRC()), (unsigned)((rows + 7) / 8), 1, 1, 256, 1, 1, 0, a1);
    void* a2[] = {&px, &pd, &pm, &pi, &pdg, &pdb, &rows, &cols};
    cuda::launch(K("b_layernorm_cols", BACKWARD_SRC()), cols * ((rows + 31) / 32), a2);
    return {dx, dg, db};
}

TensorPtr gelu_backward(const TensorPtr& x0, const TensorPtr& G0) {
    auto x = contiguous_on_device(f32(x0, "gelu")), G = contiguous_on_device(f32(G0, "gelu"));
    auto dx = empty_cuda(x->shape());
    const void* px = base(x);
    const void* pg = base(G);
    void* pd = dx->storage()->data();
    long long n = x->numel();
    void* args[] = {&px, &pg, &pd, &n};
    cuda::launch(K("b_gelu", BACKWARD_SRC()), n, args);
    return dx;
}

// Rows of W by id. `ids` is an int64 / int32 tensor on the device, or null
// with mod > 0 for positions r % mod. The op form (for recorded gradients)
// reads the ids back; a plain forward and backward never do.
TensorPtr embedding_device(const TensorPtr& W0, const TensorPtr& ids, long long mod, long long rows,
                           const Shape& out_shape) {
    auto W = contiguous_on_device(f32(W0, "Embedding"));
    TensorPtr dev_ids = ids ? contiguous_on_device(on_device(ids, "Embedding")) : nullptr;
    if (dev_ids && dev_ids->dtype() != DType::Int64 && dev_ids->dtype() != DType::Int32)
        dev_ids = cast_op(dev_ids, DType::Int64);
    long long e = W->shape().back(), vocab = W->shape()[0];
    auto out = empty_cuda(out_shape);
    const void* pw = base(W);
    const void* pi = dev_ids ? base(dev_ids) : nullptr;
    int dt = dev_ids ? (int)dev_ids->dtype() : 4;
    void* po = out->storage()->data();
    long long r = rows, md = mod;
    void* args[] = {&pw, &pi, &dt, &md, &po, &r, &e, &vocab};
    cuda::launch(K("b_embed_fwd", BACKWARD_SRC()), rows * e, args);
    if (W->requires_grad()) {
        auto node = make_node(out, {W});
        node->name = "Embedding";
        // the op form, for gradients that are themselves differentiated
        autograd::attach_reference(*node, [dev_ids, mod, rows, e, out_shape](const vector<TensorPtr>& in) -> TensorPtr {
            vector<int64_t> host((size_t)rows);
            if (dev_ids) host = index_values(dev_ids, "Embedding");
            else for (long long q = 0; q < rows; q++) host[(size_t)q] = q % mod;
            auto idx = make_shared<vector<int64_t>>((size_t)(rows * e));
            for (long long q = 0; q < rows; q++)
                for (long long d = 0; d < e; d++) (*idx)[(size_t)(q * e + d)] = host[(size_t)q] * e + d;
            return gather_flat(in[0], idx, out_shape);
        });
        auto reference = node->backward_graph;
        node->backward_graph = [reference, dev_ids, dt, mod, rows, e, vocab, W](const TensorPtr& out, const TensorPtr& G0) -> vector<TensorPtr> {
            if (grad_enabled()) return reference(out, G0);
            auto G = contiguous_on_device(f32(G0, "Embedding"));
            auto dw = make_shared<Tensor>(W->shape(), F32, Device::CUDA);      // zeroed
            const void* pg = base(G);
            const void* pi = dev_ids ? base(dev_ids) : nullptr;
            void* pd = dw->storage()->data();
            int d2 = dt;
            long long r2 = rows, ee = e, md2 = mod, vv = vocab;
            void* a[] = {&pg, &pi, &d2, &md2, &pd, &r2, &ee, &vv};
            cuda::launch(K("b_embed_bwd", BACKWARD_SRC()), r2 * ee, a);
            return {dw};
        };
    }
    return out;
}

TensorPtr embedding(const TensorPtr& W0, const vector<int64_t>& ids, const Shape& out_shape) {
    auto host = make_shared<Tensor>(Shape{(int64_t)ids.size()}, DType::Int64);
    if (!ids.empty()) memcpy(host->storage()->data(), ids.data(), ids.size() * sizeof(int64_t));
    return embedding_device(W0, host->to(Device::CUDA), 0, (long long)ids.size(), out_shape);
}

// ── fused attention ──────────────────────────────────────────────────────────
//
// softmax(scale * Q K^T + causal mask) V for every head, with Q, K, V and the
// output in [B, S, E] -- heads side by side in each row, as the projections
// produce them. Head h of sequence b starts at b*S*E + h*D with a row stride
// of E, and the H heads of one sequence are a strided batch of stride D, so
// cuBLAS reads them in place: no permute, no copies. Per sequence: one batched
// gemm for the scores, one kernel for scale + mask + softmax, one batched gemm
// for P V. The backward is four batched gemms and one kernel for the softmax
// gradient. With create_graph the op form runs instead.

static const string& ATTN_SRC() {
    static const string s = src(R"(
// rows of S scores: P = softmax(scale * s), with j > i masked when causal; in place
// One warp per row (8 rows to a 256-thread block), reductions by shuffles:
// a block per row left most of its threads idle on rows of a few hundred and
// spent its time in __syncthreads -- 4x slower than PyTorch's on a GTX 970.
__device__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// rows of S scores: P = softmax(scale * s), with j > i masked when causal; in place
extern "C" __global__ void a_softmax(float* sc, long long cols, float scale, int causal, long long rows) {
    const long long r = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int lane = threadIdx.x & 31;
    float* row = sc + r * cols;
    const long long i = r % cols;                       // the query's position
    const long long last = causal ? i : cols - 1;
    float m = -3.402823466e38f;
    for (long long j = lane; j <= last; j += 32) m = fmaxf(m, row[j] * scale);
    m = warp_max(m);
    float t = 0.0f;
    for (long long j = lane; j < cols; j += 32) {
        const float e = j <= last ? expf(row[j] * scale - m) : 0.0f;
        row[j] = e;
        t += e;
    }
    const float inv = 1.0f / warp_sum(t);
    for (long long j = lane; j < cols; j += 32) row[j] *= inv;
}
// dS = scale * P * (dP - sum(dP * P)) per row, in place on dP
extern "C" __global__ void a_softmax_back(const float* P, float* dP, long long cols, float scale, long long rows) {
    const long long r = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int lane = threadIdx.x & 31;
    const float* p = P + r * cols;
    float* d = dP + r * cols;
    float t = 0.0f;
    for (long long j = lane; j < cols; j += 32) t += d[j] * p[j];
    const float dot = warp_sum(t);
    for (long long j = lane; j < cols; j += 32) d[j] = scale * p[j] * (d[j] - dot);
}
)");
    return s;
}

TensorPtr attention(const TensorPtr& q0, const TensorPtr& k0, const TensorPtr& v0, int heads, bool causal) {
    auto q = contiguous_on_device(f32(q0, "attention")), k = contiguous_on_device(f32(k0, "attention"));
    auto v = contiguous_on_device(f32(v0, "attention"));
    if (q->dim() != 3 || q->shape() != k->shape() || q->shape() != v->shape())
        throw runtime_error("attention: q, k and v must all be [batch, seq, embed]");
    const int64_t B = q->size(0), S = q->size(1), E = q->size(2), H = heads, D = E / H;
    if (D * H != E) throw runtime_error("attention: embed must divide by heads");
    const float scale = 1.0f / std::sqrt((float)D);
    auto P = empty_cuda({B, H, S, S});
    auto O = empty_cuda({B, S, E});
    const float* pq = static_cast<const float*>(base(q));
    const float* pk = static_cast<const float*>(base(k));
    const float* pv = static_cast<const float*>(base(v));
    float* pp = static_cast<float*>(P->storage()->data());
    float* po = static_cast<float*>(O->storage()->data());
    const int s_ = (int)S, d_ = (int)D, e_ = (int)E, h_ = (int)H;
    for (int64_t b = 0; b < B; b++)       // scores = Q K^T, every head of sequence b
        cuda::sgemm_batched(false, true, s_, s_, d_, 1.0f, pq + b * S * E, e_, D, pk + b * S * E, e_, D,
                            0.0f, pp + b * H * S * S, s_, S * S, h_);
    long long cols = S;
    int cz = causal ? 1 : 0;
    float sc = scale;
    long long nrows = B * H * S;
    void* a1[] = {&pp, &cols, &sc, &cz, &nrows};
    if (B * H * S) cuda::launch(K("a_softmax", ATTN_SRC()), (unsigned)((B * H * S + 7) / 8), 1, 1, 256, 1, 1, 0, a1);
    for (int64_t b = 0; b < B; b++)       // O = P V, written into the heads' columns
        cuda::sgemm_batched(false, false, s_, d_, s_, 1.0f, pp + b * H * S * S, s_, S * S, pv + b * S * E, e_, D,
                            0.0f, po + b * S * E, e_, D, h_);

    if (q->requires_grad() || k->requires_grad() || v->requires_grad()) {
        auto node = make_node(O, {q, k, v});
        node->name = "attention";
        // the op form: what higher-order gradients differentiate
        autograd::attach_reference(*node, [B, S, E, H, D, scale, causal](const vector<TensorPtr>& in) -> TensorPtr {
            auto heads_of = [&](const TensorPtr& t) { return permute(reshape(t, {B, S, H, D}), {0, 2, 1, 3}); };
            auto scores = multiply(scale, ::matmul(heads_of(in[0]), transpose(heads_of(in[1]))));
            if (causal) {
                vector<float> m((size_t)(S * S), 0.0f);
                for (int64_t i = 0; i < S; i++) for (int64_t j = i + 1; j < S; j++) m[(size_t)(i * S + j)] = -1e9f;
                scores = add(scores, make_shared<Tensor>(Shape{S, S}, m)->to(Device::CUDA));
            }
            auto o = ::matmul(::softmax(scores), heads_of(in[2]));
            return reshape(permute(o, {0, 2, 1, 3}), {B, S, E});
        });
        auto reference = node->backward_graph;
        node->backward_graph = [reference, q, k, v, P, B, S, E, H, D, scale](const TensorPtr& out,
                                                                           const TensorPtr& G0) -> vector<TensorPtr> {
            if (grad_enabled()) return reference(out, G0);
            auto G = contiguous_on_device(f32(G0, "attention"));
            auto dP = empty_cuda({B, H, S, S});
            auto dQ = empty_cuda({B, S, E}), dK = empty_cuda({B, S, E}), dV = empty_cuda({B, S, E});
            const float* pq = static_cast<const float*>(base(q));
            const float* pk = static_cast<const float*>(base(k));
            const float* pv = static_cast<const float*>(base(v));
            const float* pp = static_cast<const float*>(base(P));
            const float* pg = static_cast<const float*>(base(G));
            float* pdp = static_cast<float*>(dP->storage()->data());
            float* pdq = static_cast<float*>(dQ->storage()->data());
            float* pdk = static_cast<float*>(dK->storage()->data());
            float* pdv = static_cast<float*>(dV->storage()->data());
            const int s_ = (int)S, d_ = (int)D, e_ = (int)E, h_ = (int)H;
            for (int64_t b = 0; b < B; b++) {
                const int64_t ob = b * S * E, pb = b * H * S * S;
                // dV = P^T dO;  dP = dO V^T
                cuda::sgemm_batched(true, false, s_, d_, s_, 1.0f, pp + pb, s_, S * S, pg + ob, e_, D,
                                    0.0f, pdv + ob, e_, D, h_);
                cuda::sgemm_batched(false, true, s_, s_, d_, 1.0f, pg + ob, e_, D, pv + ob, e_, D,
                                    0.0f, pdp + pb, s_, S * S, h_);
            }
            long long cols = S;
            float sc = scale;
            long long nrows = B * H * S;
            void* a2[] = {&pp, &pdp, &cols, &sc, &nrows};
            if (B * H * S) cuda::launch(K("a_softmax_back", ATTN_SRC()), (unsigned)((B * H * S + 7) / 8), 1, 1, 256, 1, 1, 0, a2);
            for (int64_t b = 0; b < B; b++) {
                const int64_t ob = b * S * E, pb = b * H * S * S;
                // dQ = dS K;  dK = dS^T Q
                cuda::sgemm_batched(false, false, s_, d_, s_, 1.0f, pdp + pb, s_, S * S, pk + ob, e_, D,
                                    0.0f, pdq + ob, e_, D, h_);
                cuda::sgemm_batched(true, false, s_, d_, s_, 1.0f, pdp + pb, s_, S * S, pq + ob, e_, D,
                                    0.0f, pdk + ob, e_, D, h_);
            }
            return {q->requires_grad() ? dQ : nullptr, k->requires_grad() ? dK : nullptr,
                    v->requires_grad() ? dV : nullptr};
        };
    }
    return O;
}

// Attention straight from one fused projection: qkv [B, S, 3E] holds Q, K and
// V side by side in each row (x @ [W_q W_k W_v], a single gemm), and every
// head is read in place at row stride 3E. With RoPE, one kernel writes the
// rotated Q and K (positions restart at 0 in each sequence) and the scores use
// those; the backward rotates their gradients back in place. The gradient is
// a single [B, S, 3E] tensor laid out like qkv.

static const string& ROPE_SRC() {
    static const string s = src(R"(
// pair i of head h at row (b, t): rotate (v[2i], v[2i+1]) by t * 10000^(-2i/D).
// sign +1 rotates (forward), -1 the inverse (backward). src / dst row strides.
__device__ void rope_pair(float* c, float* s, int t, int i, int D, float sign) {
    const float theta = powf(10000.0f, -2.0f * (float)i / (float)D);
    const float angle = (float)t * theta;
    *c = cosf(angle); *s = sign * sinf(angle);
}
extern "C" __global__ void a_rope(const float* qkv, float* Q, float* K, long long rows, int S, int E, int D) {
    long long n = gid(); const int half = E / 2; if (n >= rows * half) return;
    const long long r = n / half; const int p = (int)(n % half);         // pair p of the row's E dims
    const int h = p / (D / 2), i = p % (D / 2), t = (int)(r % S);
    const long long col = (long long)h * D + 2 * i;
    float c, s; rope_pair(&c, &s, t, i, D, 1.0f);
    const float* src = qkv + r * 3 * E;
    const float q0 = src[col], q1 = src[col + 1], k0 = src[E + col], k1 = src[E + col + 1];
    Q[r * E + col] = q0 * c - q1 * s;  Q[r * E + col + 1] = q0 * s + q1 * c;
    K[r * E + col] = k0 * c - k1 * s;  K[r * E + col + 1] = k0 * s + k1 * c;
}
// pointers for a two-level batch: matrix (b, h) of X starts at
// X + b * sb + h * sh; three operands, B * H matrices, written into ptr[3][B*H]
extern "C" __global__ void a_ptrs(const float* x, long long xb, long long xh, const float* y, long long yb,
                                  long long yh, float* z, long long zb, long long zh, int H, int n,
                                  unsigned long long* ptr) {
    int i = (int)gid(); if (i >= n) return;
    const long long b = i / H, h = i % H;
    ptr[i] = (unsigned long long)(x + b * xb + h * xh);
    ptr[n + i] = (unsigned long long)(y + b * yb + h * yh);
    ptr[2 * n + i] = (unsigned long long)(z + b * zb + h * zh);
}
// the inverse rotation, in place on the Q and K thirds of dqkv [rows, 3E]
extern "C" __global__ void a_rope_back(float* g, long long rows, int S, int E, int D) {
    long long n = gid(); const int half = E / 2; if (n >= rows * half) return;
    const long long r = n / half; const int p = (int)(n % half);
    const int h = p / (D / 2), i = p % (D / 2), t = (int)(r % S);
    const long long col = (long long)h * D + 2 * i;
    float c, s; rope_pair(&c, &s, t, i, D, -1.0f);
    float* row = g + r * 3 * E;
    for (int part = 0; part < 2; part++) {
        float* v = row + part * E + col;
        const float a = v[0], b = v[1];
        v[0] = a * c - b * s; v[1] = a * s + b * c;
    }
}
)");
    return s;
}

// One cuBLAS call for every (sequence, head) pair: matrix (b, h) of each
// operand starts at base + b * stride_b + h * stride_h, a layout no single
// stride describes. Calling per sequence left 4 small products per call --
// too few to fill the GPU, 192 calls a GPT step.
struct Op2 { const float* p; long long sb, sh; int ld; };
static void gemm_bh(bool ta, bool tb, int m, int n, int k, Op2 a, Op2 b, float beta, float* c, long long csb,
                    long long csh, int ldc, int B, int H) {
    const int nb = B * H;
    if (!nb) return;
    auto ptr = empty_cuda({3LL * nb}, DType::Int64);
    unsigned long long* pp = static_cast<unsigned long long*>(ptr->storage()->data());
    const float* x = a.p;
    const float* y = b.p;
    float* z = c;
    long long xb = a.sb, xh = a.sh, yb = b.sb, yh = b.sh, zb = csb, zh = csh;
    int h_ = H, n_ = nb;
    void* args[] = {&x, &xb, &xh, &y, &yb, &yh, &z, &zb, &zh, &h_, &n_, &pp};
    cuda::launch(K("a_ptrs", ROPE_SRC()), nb, args);
    auto P = reinterpret_cast<const float* const*>(pp);
    cuda::sgemm_pointers(ta, tb, m, n, k, 1.0f, P, a.ld, P + nb, b.ld, beta,
                         reinterpret_cast<float* const*>(pp + 2 * nb), ldc, nb);
}

TensorPtr attention_qkv(const TensorPtr& qkv0, int heads, bool causal, bool rope) {
    auto qkv = contiguous_on_device(f32(qkv0, "attention"));
    if (qkv->dim() != 3 || qkv->size(2) % 3 != 0)
        throw runtime_error("attention: qkv must be [batch, seq, 3 * embed]");
    const int64_t B = qkv->size(0), S = qkv->size(1), E = qkv->size(2) / 3, H = heads, D = E / H;
    if (D * H != E) throw runtime_error("attention: embed must divide by heads");
    if (rope && D % 2) throw runtime_error("attention: RoPE needs an even head size");
    const float scale = 1.0f / std::sqrt((float)D);
    const int64_t E3 = 3 * E, rows = B * S;
    const float* px = static_cast<const float*>(base(qkv));
    // Q and K as the scores read them: in place in qkv, or rotated copies
    TensorPtr Qr, Kr;
    const float *pq = px, *pk = px + E;
    int64_t lqk = E3;
    if (rope) {
        Qr = empty_cuda({B, S, E});
        Kr = empty_cuda({B, S, E});
        float* q = static_cast<float*>(Qr->storage()->data());
        float* k = static_cast<float*>(Kr->storage()->data());
        long long rr = rows;
        int s_ = (int)S, e_ = (int)E, d_ = (int)D;
        void* a[] = {&px, &q, &k, &rr, &s_, &e_, &d_};
        cuda::launch(K("a_rope", ROPE_SRC()), rows * (E / 2), a);
        pq = q; pk = k; lqk = E;
    }
    const float* pv = px + 2 * E;
    auto P = empty_cuda({B, H, S, S});
    auto O = empty_cuda({B, S, E});
    float* pp = static_cast<float*>(P->storage()->data());
    float* po = static_cast<float*>(O->storage()->data());
    const int s_ = (int)S, d_ = (int)D, e_ = (int)E, h_ = (int)H, lq = (int)lqk, l3 = (int)E3;
    // scores = Q K^T for every head of every sequence, one call
    gemm_bh(false, true, s_, s_, d_, {pq, S * lqk, D, lq}, {pk, S * lqk, D, lq}, 0.0f, pp, H * S * S, S * S, s_,
            (int)B, h_);
    long long cols = S;
    int cz = causal ? 1 : 0;
    float sc = scale;
    long long nrows = B * H * S;
    void* a1[] = {&pp, &cols, &sc, &cz, &nrows};
    if (B * H * S) cuda::launch(K("a_softmax", ATTN_SRC()), (unsigned)((B * H * S + 7) / 8), 1, 1, 256, 1, 1, 0, a1);
    // O = P V, written into the heads' columns
    gemm_bh(false, false, s_, d_, s_, {pp, H * S * S, S * S, s_}, {pv, S * E3, D, l3}, 0.0f, po, S * E, D, e_,
            (int)B, h_);

    if (qkv->requires_grad()) {
        auto node = make_node(O, {qkv});
        node->name = "attention";
        // the op form: what higher-order gradients differentiate
        autograd::attach_reference(*node, [B, S, E, H, D, scale, causal, rope](const vector<TensorPtr>& in) -> TensorPtr {
            auto part = [&](int k) {           // [B, S, E] third k of the rows
                auto idx = make_shared<vector<int64_t>>((size_t)(B * S * E));
                for (int64_t r = 0; r < B * S; r++)
                    for (int64_t j = 0; j < E; j++) (*idx)[(size_t)(r * E + j)] = r * 3 * E + k * E + j;
                return gather_flat(in[0], idx, Shape{B, S, E});
            };
            auto heads_of = [&](const TensorPtr& t) { return permute(reshape(t, {B, S, H, D}), {0, 2, 1, 3}); };
            auto Q = heads_of(part(0)), Kh = heads_of(part(1)), V = heads_of(part(2));
            if (rope) {
                // a constant per-position rotation: v' = v @ R_t
                vector<float> R((size_t)(S * D * D), 0.0f);
                for (int64_t t = 0; t < S; t++)
                    for (int64_t i = 0; i < D / 2; i++) {
                        const float theta = powf(10000.0f, -2.0f * (float)i / (float)D), angle = (float)t * theta;
                        const float c = cosf(angle), sn = sinf(angle);
                        float* r = R.data() + t * D * D;
                        r[(2 * i) * D + 2 * i] = c;       r[(2 * i) * D + 2 * i + 1] = sn;
                        r[(2 * i + 1) * D + 2 * i] = -sn; r[(2 * i + 1) * D + 2 * i + 1] = c;
                    }
                auto Rt = make_shared<Tensor>(Shape{S, D, D}, R)->to(Device::CUDA);
                Q = reshape(::matmul(reshape(Q, {B, H, S, 1, D}), Rt), {B, H, S, D});
                Kh = reshape(::matmul(reshape(Kh, {B, H, S, 1, D}), Rt), {B, H, S, D});
            }
            auto scores = multiply(scale, ::matmul(Q, transpose(Kh)));
            if (causal) {
                vector<float> m((size_t)(S * S), 0.0f);
                for (int64_t i = 0; i < S; i++) for (int64_t j = i + 1; j < S; j++) m[(size_t)(i * S + j)] = -1e9f;
                scores = add(scores, make_shared<Tensor>(Shape{S, S}, m)->to(Device::CUDA));
            }
            auto o = ::matmul(::softmax(scores), V);
            return reshape(permute(o, {0, 2, 1, 3}), {B, S, E});
        });
        auto reference = node->backward_graph;
        node->backward_graph = [reference, qkv, Qr, Kr, P, B, S, E, H, D, scale, rope](
                                   const TensorPtr& out, const TensorPtr& G0) -> vector<TensorPtr> {
            if (grad_enabled()) return reference(out, G0);
            auto G = contiguous_on_device(f32(G0, "attention"));
            const int64_t E3 = 3 * E;
            auto dP = empty_cuda({B, H, S, S});
            auto dX = empty_cuda({B, S, E3});
            const float* px = static_cast<const float*>(base(qkv));
            const float* pq = rope ? static_cast<const float*>(base(Qr)) : px;
            const float* pk = rope ? static_cast<const float*>(base(Kr)) : px + E;
            const int64_t lqk = rope ? E : E3;
            const float* pv = px + 2 * E;
            const float* pp = static_cast<const float*>(base(P));
            const float* pg = static_cast<const float*>(base(G));
            float* pdp = static_cast<float*>(dP->storage()->data());
            float* pdq = static_cast<float*>(dX->storage()->data());
            float* pdk = pdq + E;
            float* pdv = pdq + 2 * E;
            const int s_ = (int)S, d_ = (int)D, e_ = (int)E, h_ = (int)H, lq = (int)lqk, l3 = (int)E3;
            // dV = P^T dO;  dP = dO V^T  -- each one call over every (sequence, head)
            gemm_bh(true, false, s_, d_, s_, {pp, H * S * S, S * S, s_}, {pg, S * E, D, e_}, 0.0f, pdv, S * E3, D,
                    l3, (int)B, h_);
            gemm_bh(false, true, s_, s_, d_, {pg, S * E, D, e_}, {pv, S * E3, D, l3}, 0.0f, pdp, H * S * S, S * S,
                    s_, (int)B, h_);
            long long cols = S;
            float sc = scale;
            long long nrows = B * H * S;
            void* a2[] = {&pp, &pdp, &cols, &sc, &nrows};
            if (B * H * S) cuda::launch(K("a_softmax_back", ATTN_SRC()), (unsigned)((B * H * S + 7) / 8), 1, 1, 256, 1, 1, 0, a2);
            // dQ = dS K;  dK = dS^T Q   (in the rotated space under RoPE)
            gemm_bh(false, false, s_, d_, s_, {pdp, H * S * S, S * S, s_}, {pk, S * lqk, D, lq}, 0.0f, pdq, S * E3,
                    D, l3, (int)B, h_);
            gemm_bh(true, false, s_, d_, s_, {pdp, H * S * S, S * S, s_}, {pq, S * lqk, D, lq}, 0.0f, pdk, S * E3,
                    D, l3, (int)B, h_);
            if (rope) {
                long long rr = B * S;
                int s2 = (int)S, e2 = (int)E, d2 = (int)D;
                void* a3[] = {&pdq, &rr, &s2, &e2, &d2};
                cuda::launch(K("a_rope_back", ROPE_SRC()), B * S * (E / 2), a3);
            }
            return {dX};
        };
    }
    return O;
}

// ── fused transformer pieces: bias + GELU, bias + residual, residual + LayerNorm ──
//
// What a transformer block does between its gemms, each in one kernel instead
// of two or three. Gradients as for the other fused kernels: one kernel for a
// plain backward, the op form when the gradient is itself recorded.

static const string& BLOCK_SRC() {
    static const string s = src(R"(
__device__ float gelu_f(float v) { return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))); }
__device__ float gelu_d(float v) {
    const float k = 0.7978845608f, c = 0.044715f, t = tanhf(k * (v + c * v * v * v));
    return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * k * (1.0f + 3.0f * c * v * v);
}
// y = gelu(z + b), b broadcast along the last dim
extern "C" __global__ void k_bias_gelu(const float* z, const float* b, float* y, long long n, long long cols) {
    long long i = gid(); if (i >= n) return;
    y[i] = gelu_f(z[i] + b[i % cols]);
}
extern "C" __global__ void k_bias_gelu_back(const float* z, const float* b, const float* dy, float* dz,
                                            long long n, long long cols) {
    long long i = gid(); if (i >= n) return;
    dz[i] = dy[i] * gelu_d(z[i] + b[i % cols]);
}
// out = y + b + r, b broadcast along the last dim
extern "C" __global__ void k_add_bias_residual(const float* y, const float* b, const float* r, float* out,
                                               long long n, long long cols) {
    long long i = gid(); if (i >= n) return;
    out[i] = y[i] + b[i % cols] + r[i];
}
// s = a + c and y = LayerNorm(s) per row, one block per row
extern "C" __global__ void k_add_layernorm(const float* a, const float* c, float* s_out, const float* g,
                                           const float* bt, float* y, long long cols, float eps) {
    __shared__ float sh[256];
    const long long off = (long long)blockIdx.x * cols;
    float t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) { float v = a[off + j] + c[off + j]; s_out[off + j] = v; t += v; }
    sh[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w]; __syncthreads(); }
    const float mean = sh[0] / (float)cols;
    __syncthreads();
    t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) { float d = s_out[off + j] - mean; t += d * d; }
    sh[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w]; __syncthreads(); }
    const float inv = rsqrtf(sh[0] / (float)cols + eps);
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) y[off + j] = (s_out[off + j] - mean) * inv * g[j] + bt[j];
}
)");
    return s;
}

TensorPtr bias_gelu(const TensorPtr& z0, const TensorPtr& b0) {
    auto z = contiguous_on_device(f32(z0, "bias_gelu")), b = contiguous_on_device(f32(b0, "bias_gelu"));
    long long cols = z->shape().back(), n = z->numel();
    if (b->numel() != cols) throw runtime_error("bias_gelu: bias must have " + to_string(cols) + " elements");
    auto y = empty_cuda(z->shape());
    const void* pz = base(z);
    const void* pb = base(b);
    void* py = y->storage()->data();
    void* args[] = {&pz, &pb, &py, &n, &cols};
    cuda::launch(K("k_bias_gelu", BLOCK_SRC()), n, args);
    if (z->requires_grad() || b->requires_grad()) {
        auto node = make_node(y, {z, b});
        node->name = "bias_gelu";
        autograd::attach_reference(*node, [](const vector<TensorPtr>& in) -> TensorPtr { return gelu(add(in[0], in[1])); });
        auto reference = node->backward_graph;
        node->backward_graph = [reference, z, b, n, cols](const TensorPtr& out, const TensorPtr& G0) -> vector<TensorPtr> {
            if (grad_enabled()) return reference(out, G0);
            auto G = contiguous_on_device(f32(G0, "bias_gelu"));
            auto dz = empty_cuda(z->shape());
            const void* pz = base(z);
            const void* pb = base(b);
            const void* pg = base(G);
            void* pd = dz->storage()->data();
            long long nn = n, cc = cols;
            void* a[] = {&pz, &pb, &pg, &pd, &nn, &cc};
            cuda::launch(K("k_bias_gelu_back", BLOCK_SRC()), nn, a);
            TensorPtr db = b->requires_grad() ? ::sum(reshape(dz, {n / cols, cols}), 0) : nullptr;
            return {z->requires_grad() ? dz : nullptr, db};
        };
    }
    return y;
}

TensorPtr add_bias_residual(const TensorPtr& y0, const TensorPtr& b0, const TensorPtr& r0) {
    auto y = contiguous_on_device(f32(y0, "add_bias_residual")), b = contiguous_on_device(f32(b0, "add_bias_residual"));
    auto r = contiguous_on_device(f32(r0, "add_bias_residual"));
    long long cols = y->shape().back(), n = y->numel();
    if (b->numel() != cols || r->shape() != y->shape())
        throw runtime_error("add_bias_residual: bias must be [cols] and the residual the shape of y");
    auto out = empty_cuda(y->shape());
    const void* py = base(y);
    const void* pb = base(b);
    const void* pr = base(r);
    void* po = out->storage()->data();
    void* args[] = {&py, &pb, &pr, &po, &n, &cols};
    cuda::launch(K("k_add_bias_residual", BLOCK_SRC()), n, args);
    attach(out, {y, b, r}, "add_bias_residual", [y, b, r](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        return {y->requires_grad() ? G : nullptr, b->requires_grad() ? sum_to(G, b->shape()) : nullptr,
                r->requires_grad() ? G : nullptr};
    });
    return out;
}

pair<TensorPtr, TensorPtr> add_layernorm(const TensorPtr& a0, const TensorPtr& c0, const TensorPtr& g0,
                                         const TensorPtr& bt0, float eps) {
    auto a = contiguous_on_device(f32(a0, "add_layernorm")), c = contiguous_on_device(f32(c0, "add_layernorm"));
    auto g = contiguous_on_device(f32(g0, "add_layernorm")), bt = contiguous_on_device(f32(bt0, "add_layernorm"));
    if (a->shape() != c->shape()) throw runtime_error("add_layernorm: the two inputs differ in shape");
    long long cols = a->shape().back();
    const long long rows = cols ? a->numel() / cols : 0;
    auto s = empty_cuda(a->shape()), y = empty_cuda(a->shape());
    const void* pa = base(a);
    const void* pc = base(c);
    const void* pg = base(g);
    const void* pbt = base(bt);
    void* ps = s->storage()->data();
    void* py = y->storage()->data();
    void* args[] = {&pa, &pc, &ps, &pg, &pbt, &py, &cols, &eps};
    if (rows) cuda::launch(K("k_add_layernorm", BLOCK_SRC()), (unsigned)rows, 1, 1, 256, 1, 1, 0, args);
    // s is a + c; y is LayerNorm(s), with the LayerNorm gradient: fused, or the op form
    attach(s, {a, c}, "add", [a, c](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        return {a->requires_grad() ? G : nullptr, c->requires_grad() ? G : nullptr};
    });
    if (s->requires_grad() || g->requires_grad() || bt->requires_grad()) {
        auto node = make_node(y, {s, g, bt});
        node->name = "LayerNorm::forward";
        autograd::attach_reference(*node, [eps](const vector<TensorPtr>& in) -> TensorPtr {
            const int last = in[0]->dim() - 1;
            auto m = mean(in[0], last, true);
            auto d = subtract(in[0], m);
            auto var = mean(multiply(d, d), last, true);
            return add(multiply(divide(d, sqrt_op(add(eps, var))), in[1]), in[2]);
        });
        auto reference = node->backward_graph;
        node->backward_graph = [reference, s, g, bt, eps](const TensorPtr& out, const TensorPtr& G) -> vector<TensorPtr> {
            if (grad_enabled()) return reference(out, G);
            auto r = layernorm_backward(s, g, G, eps);
            return {s->requires_grad() ? r[0] : nullptr, g->requires_grad() ? r[1] : nullptr,
                    bt->requires_grad() ? r[2] : nullptr};
        };
    }
    return {s, y};
}

// ── convolution and pooling on the device ────────────────────────────────────
//
// The standard non-cuDNN route: im2col turns every output position's patch
// into a column, so a convolution is one batched gemm per call; col2im is its
// adjoint (a gather here, so no atomics and a deterministic sum). Pooling keeps
// each window's winner on the device. Gradients: kernels for a plain backward,
// the layer's op-form reference when the gradient is itself recorded.

static const string& VISION_SRC() {
    static const string s = src(R"(
// col[b, (c*K + kh)*K + kw, oy*Wc + ox] = x[b, c, oy*s + kh - p, ox*s + kw - p] (0 outside)
extern "C" __global__ void v_im2col(const float* x, float* col, long long B, long long C, long long H, long long W,
                                    int K, int s, int p, long long Hc, long long Wc) {
    const long long HW = Hc * Wc, CKK = C * K * K;
    long long i = gid(); if (i >= B * CKK * HW) return;
    const long long pos = i % HW, r = (i / HW) % CKK, b = i / (HW * CKK);
    const long long oy = pos / Wc, ox = pos % Wc;
    const int kw = (int)(r % K), kh = (int)((r / K) % K);
    const long long c = r / (K * K);
    const long long y = oy * s + kh - p, xx = ox * s + kw - p;
    col[i] = (y < 0 || y >= H || xx < 0 || xx >= W) ? 0.0f : x[((b * C + c) * H + y) * W + xx];
}
// the adjoint: x[b, c, y, xx] = sum over the (kh, kw) whose window covers it
extern "C" __global__ void v_col2im(const float* col, float* x, long long B, long long C, long long H, long long W,
                                    int K, int s, int p, long long Hc, long long Wc) {
    long long i = gid(); if (i >= B * C * H * W) return;
    const long long xx = i % W, y = (i / W) % H, c = (i / (W * H)) % C, b = i / (W * H * C);
    const long long HW = Hc * Wc, CKK = C * K * K;
    float acc = 0.0f;
    for (int kh = 0; kh < K; kh++) {
        const long long ty = y + p - kh;
        if (ty < 0 || ty % s) continue;
        const long long oy = ty / s;
        if (oy >= Hc) continue;
        for (int kw = 0; kw < K; kw++) {
            const long long tx = xx + p - kw;
            if (tx < 0 || tx % s) continue;
            const long long ox = tx / s;
            if (ox >= Wc) continue;
            acc += col[(b * CKK + (c * K + kh) * K + kw) * HW + oy * Wc + ox];
        }
    }
    x[i] = acc;
}
// The same two with the batch folded into the columns: col[r, b*HW + pos], a
// [C*K*K, B*HW] matrix. Then a convolution over the whole batch is one gemm
// rather than one per image.
extern "C" __global__ void v_im2col_b(const float* x, float* col, long long B, long long C, long long H, long long W,
                                      int K, int s, int p, long long Hc, long long Wc) {
    const long long HW = Hc * Wc, CKK = C * K * K, BHW = B * HW;
    long long i = gid(); if (i >= CKK * BHW) return;
    const long long r = i / BHW, q = i % BHW, b = q / HW, pos = q % HW;
    const long long oy = pos / Wc, ox = pos % Wc;
    const int kw = (int)(r % K), kh = (int)((r / K) % K);
    const long long c = r / (K * K);
    const long long y = oy * s + kh - p, xx = ox * s + kw - p;
    col[i] = (y < 0 || y >= H || xx < 0 || xx >= W) ? 0.0f : x[((b * C + c) * H + y) * W + xx];
}
extern "C" __global__ void v_col2im_b(const float* col, float* x, long long B, long long C, long long H, long long W,
                                      int K, int s, int p, long long Hc, long long Wc) {
    long long i = gid(); if (i >= B * C * H * W) return;
    const long long xx = i % W, y = (i / W) % H, c = (i / (W * H)) % C, b = i / (W * H * C);
    const long long HW = Hc * Wc, BHW = B * HW;
    float acc = 0.0f;
    for (int kh = 0; kh < K; kh++) {
        const long long ty = y + p - kh;
        if (ty < 0 || ty % s) continue;
        const long long oy = ty / s;
        if (oy >= Hc) continue;
        for (int kw = 0; kw < K; kw++) {
            const long long tx = xx + p - kw;
            if (tx < 0 || tx % s) continue;
            const long long ox = tx / s;
            if (ox >= Wc) continue;
            acc += col[((c * K + kh) * K + kw) * BHW + b * HW + oy * Wc + ox];
        }
    }
    x[i] = acc;
}
// [C, B*P] -> [B, C, P], adding bias[c] if given; and [B, C, P] -> [C, B*P]
extern "C" __global__ void v_from_cb(const float* y, const float* bias, float* out, long long B, long long C, long long P) {
    long long i = gid(); if (i >= B * C * P) return;
    const long long pos = i % P, c = (i / P) % C, b = i / (P * C);
    out[i] = y[c * B * P + b * P + pos] + (bias ? bias[c] : 0.0f);
}
extern "C" __global__ void v_to_cb(const float* x, float* out, long long B, long long C, long long P) {
    long long i = gid(); if (i >= B * C * P) return;
    const long long q = i % (B * P), c = i / (B * P), b = q / P, pos = q % P;
    out[i] = x[(b * C + c) * P + pos];
}
// y[b, c, pos] += bias[c], in place
extern "C" __global__ void v_add_bias_c(float* y, const float* bias, long long C, long long P, long long n) {
    long long i = gid(); if (i >= n) return;
    y[i] += bias[(i / P) % C];
}
// max pooling: value and the flat input index of each window's winner (first max)
extern "C" __global__ void v_maxpool(const float* x, float* y, long long* arg, long long planes, long long H, long long W,
                                     int K, int s, long long Ho, long long Wo) {
    long long i = gid(); if (i >= planes * Ho * Wo) return;
    const long long ox = i % Wo, oy = (i / Wo) % Ho, pl = i / (Wo * Ho);
    const float* xp = x + pl * H * W;
    long long best = (oy * s) * W + ox * s;
    float bv = xp[best];
    for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) {
            const long long j = (oy * s + kh) * W + ox * s + kw;
            if (xp[j] > bv) { bv = xp[j]; best = j; }
        }
    y[i] = bv;
    arg[i] = pl * H * W + best;
}
extern "C" __global__ void v_scatter_rows(const float* g, const long long* idx, float* dx, long long n) {
    long long i = gid(); if (i >= n) return;
    atomicAdd(&dx[idx[i]], g[i]);
}
// max-pool backward when windows do not overlap (stride >= K): each input
// element is in at most one window, so it reads that window's gradient if it
// was the winner and writes 0 otherwise -- every element written once, with
// no zero-fill and no atomics
extern "C" __global__ void v_maxpool_back(const float* g, const long long* arg, float* dx, long long planes,
                                          long long H, long long W, int K, int s, long long Ho, long long Wo) {
    long long i = gid(); if (i >= planes * H * W) return;
    const long long xx = i % W, y = (i / W) % H, pl = i / (W * H);
    const long long oy = y / s, ox = xx / s;
    float r = 0.0f;
    if (oy < Ho && ox < Wo && y - oy * s < K && xx - ox * s < K) {
        const long long o = (pl * Ho + oy) * Wo + ox;
        if (arg[o] == i) r = g[o];
    }
    dx[i] = r;
}
extern "C" __global__ void v_avgpool(const float* x, float* y, long long planes, long long H, long long W,
                                     int K, int s, long long Ho, long long Wo) {
    long long i = gid(); if (i >= planes * Ho * Wo) return;
    const long long ox = i % Wo, oy = (i / Wo) % Ho, pl = i / (Wo * Ho);
    const float* xp = x + pl * H * W;
    float acc = 0.0f;
    for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) acc += xp[(oy * s + kh) * W + ox * s + kw];
    y[i] = acc / (float)(K * K);
}
extern "C" __global__ void v_avgpool_back(const float* g, float* dx, long long planes, long long H, long long W,
                                          int K, int s, long long Ho, long long Wo) {
    long long i = gid(); if (i >= planes * Ho * Wo) return;
    const long long ox = i % Wo, oy = (i / Wo) % Ho, pl = i / (Wo * Ho);
    const float v = g[i] / (float)(K * K);
    for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) atomicAdd(&dx[pl * H * W + (oy * s + kh) * W + ox * s + kw], v);
}
)");
    return s;
}

// a kernel from VISION_SRC (named VK: these functions have a kernel size K)
static cuda::Kernel VK(const char* name) { return K(name, VISION_SRC()); }

// x [B, C, H, W] -> col [B, C*K*K, Hc*Wc]
static TensorPtr im2col(const TensorPtr& x, long long C, long long H, long long W, int K, int s, int p,
                        long long Hc, long long Wc) {
    long long B = x->size(0);
    auto col = empty_cuda({B, C * K * K, Hc * Wc});
    const void* px = base(x);
    void* pc = col->storage()->data();
    void* args[] = {&px, &pc, &B, &C, &H, &W, &K, &s, &p, &Hc, &Wc};
    cuda::launch(VK("v_im2col"), col->numel(), args);
    return col;
}

// col [B, C*K*K, Hc*Wc] -> x [B, C, H, W]
static TensorPtr col2im(const TensorPtr& col, long long B, long long C, long long H, long long W, int K, int s, int p,
                        long long Hc, long long Wc) {
    auto x = empty_cuda({B, C, H, W});
    const void* pc = base(col);
    void* px = x->storage()->data();
    void* args[] = {&pc, &px, &B, &C, &H, &W, &K, &s, &p, &Hc, &Wc};
    cuda::launch(VK("v_col2im"), x->numel(), args);
    return x;
}

// x [B, C, H, W] -> col [C*K*K, B*Hc*Wc]
static TensorPtr im2col_b(const TensorPtr& x, long long C, long long H, long long W, int K, int s, int p,
                          long long Hc, long long Wc) {
    long long B = x->size(0);
    auto col = empty_cuda({C * K * K, B * Hc * Wc});
    const void* px = base(x);
    void* pc = col->storage()->data();
    void* args[] = {&px, &pc, &B, &C, &H, &W, &K, &s, &p, &Hc, &Wc};
    cuda::launch(VK("v_im2col_b"), col->numel(), args);
    return col;
}

// col [C*K*K, B*Hc*Wc] -> x [B, C, H, W]
static TensorPtr col2im_b(const TensorPtr& col, long long B, long long C, long long H, long long W, int K, int s,
                          int p, long long Hc, long long Wc) {
    auto x = empty_cuda({B, C, H, W});
    const void* pc = base(col);
    void* px = x->storage()->data();
    void* args[] = {&pc, &px, &B, &C, &H, &W, &K, &s, &p, &Hc, &Wc};
    cuda::launch(VK("v_col2im_b"), x->numel(), args);
    return x;
}

// [C, B*P] -> [B, C, P] (+ bias[c]); [B, C, P] -> [C, B*P]
static TensorPtr from_cb(const TensorPtr& y, const TensorPtr& bias, long long B, long long C, long long P,
                         const Shape& shape) {
    auto out = empty_cuda(shape);
    const void* py = base(y);
    const void* pb = bias ? base(bias) : nullptr;
    void* po = out->storage()->data();
    void* args[] = {&py, &pb, &po, &B, &C, &P};
    cuda::launch(VK("v_from_cb"), B * C * P, args);
    return out;
}
static TensorPtr to_cb(const TensorPtr& x, long long B, long long C, long long P) {
    auto out = empty_cuda({C, B * P});
    const void* px = base(x);
    void* po = out->storage()->data();
    void* args[] = {&px, &po, &B, &C, &P};
    cuda::launch(VK("v_to_cb"), B * C * P, args);
    return out;
}

// y [B, C, P] += bias[c] in place; no bias, nothing to do
static void add_bias_c(const TensorPtr& y, const TensorPtr& bias, long long C, long long P) {
    if (!bias) return;
    long long n = y->numel();
    void* py = y->storage()->data();
    const void* pb = base(bias);
    void* args[] = {&py, &pb, &C, &P, &n};
    cuda::launch(VK("v_add_bias_c"), n, args);
}

static const float* fp(const TensorPtr& t) { return static_cast<const float*>(base(t)); }
static float* wp(const TensorPtr& t) { return static_cast<float*>(t->storage()->data()); }

// db for a [B, C, H, W] gradient: the sum over everything but the channel
static TensorPtr bias_grad(const TensorPtr& G, long long B, long long C, long long P) {
    return ::sum(::sum(reshape(G, {B, C, P}), 2), 0);
}

using Reference = std::function<TensorPtr(const vector<TensorPtr>&)>;

// a node on `out` over `inputs`: `fused` for a plain backward, `reference` (the
// layer's op form) when the gradient is recorded
static void attach_fused(const TensorPtr& out, vector<TensorPtr> inputs, const char* name, Reference reference,
                         function<vector<TensorPtr>(const TensorPtr& G)> fused) {
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

TensorPtr conv2d(const TensorPtr& x0, const TensorPtr& w0, const TensorPtr& b0, int stride, int pad, Reference reference) {
    auto x = contiguous_on_device(f32(x0, "Conv2D")), w = contiguous_on_device(f32(w0, "Conv2D"));
    auto bias = contiguous_on_device(f32(b0, "Conv2D"));
    const long long B = x->size(0), C = x->size(1), H = x->size(2), W = x->size(3);
    const long long F = w->size(0);
    const int K = (int)w->size(2);
    const long long Ho = (H + 2 * pad - K) / stride + 1, Wo = (W + 2 * pad - K) / stride + 1;
    const long long HW = Ho * Wo, CKK = C * K * K, BHW = B * HW;
    if (cuda::dnn_available()) {
        const cuda::ConvShape cs{(int)B, (int)C, (int)H, (int)W, (int)F, K, stride, pad};
        auto out = empty_cuda({B, F, Ho, Wo});
        cuda::dnn_conv_forward(cs, fp(x), fp(w), wp(out));
        add_bias_c(out, bias, F, HW);
        attach_fused(out, {x0, w0, b0}, "Conv2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
            auto G = contiguous_on_device(f32(G0, "Conv2D"));
            auto dx = empty_cuda({B, C, H, W});
            auto dw = empty_cuda({F, C, (long long)K, (long long)K});
            cuda::dnn_conv_backward_data(cs, fp(w), fp(G), wp(dx));
            cuda::dnn_conv_backward_filter(cs, fp(x), fp(G), wp(dw));
            return {dx, dw, bias_grad(G, B, F, HW)};
        });
        return out;
    }
    // the whole batch as one gemm: W [F, CKK] @ col [CKK, B*HW] -> [F, B*HW],
    // then back to [B, F, Ho, Wo] with the bias added in the same pass
    auto col = im2col_b(x, C, H, W, K, stride, pad, Ho, Wo);
    auto yf = empty_cuda({F, BHW});
    cuda::sgemm(false, false, (int)F, (int)BHW, (int)CKK, 1.0f, static_cast<const float*>(base(w)), (int)CKK,
                static_cast<const float*>(base(col)), (int)BHW, 0.0f, static_cast<float*>(yf->storage()->data()), (int)BHW);
    auto out = from_cb(yf, bias, B, F, HW, {B, F, Ho, Wo});
    attach_fused(out, {x0, w0, b0}, "Conv2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "Conv2D"));
        auto gf = to_cb(G, B, F, HW);                                   // [F, B*HW]
        const float* pg = static_cast<const float*>(base(gf));
        // dW = G' @ col^T: one gemm over every image's positions
        auto dw = empty_cuda({F, C, (long long)K, (long long)K});
        cuda::sgemm(false, true, (int)F, (int)CKK, (int)BHW, 1.0f, pg, (int)BHW, static_cast<const float*>(base(col)),
                    (int)BHW, 0.0f, static_cast<float*>(dw->storage()->data()), (int)CKK);
        // dcol = W^T @ G', then back onto the input
        auto dcol = empty_cuda({CKK, BHW});
        cuda::sgemm(true, false, (int)CKK, (int)BHW, (int)F, 1.0f, static_cast<const float*>(base(w)), (int)CKK, pg,
                    (int)BHW, 0.0f, static_cast<float*>(dcol->storage()->data()), (int)BHW);
        auto dx = col2im_b(dcol, B, C, H, W, K, stride, pad, Ho, Wo);
        auto db = ::sum(gf, 1);
        return {dx, dw, db};
    });
    return out;
}

TensorPtr conv_transpose2d(const TensorPtr& x0, const TensorPtr& w0, const TensorPtr& b0, int stride, int pad,
                           Reference reference) {
    auto x = contiguous_on_device(f32(x0, "ConvTranspose2D")), w = contiguous_on_device(f32(w0, "ConvTranspose2D"));
    auto bias = contiguous_on_device(f32(b0, "ConvTranspose2D"));
    const long long B = x->size(0), Cin = x->size(1), H = x->size(2), W = x->size(3);
    const long long Cout = w->size(1);
    const int K = (int)w->size(2);
    const long long Ho = (H - 1) * stride - 2 * pad + K, Wo = (W - 1) * stride - 2 * pad + K;
    const long long HWi = H * W, CKK = Cout * K * K, BHW = B * HWi;
    if (cuda::dnn_available()) {
        // the transpose of a convolution from [B, Cout, Ho, Wo] to [B, Cin, H, W]
        // with filter [Cin, Cout, K, K]: its forward is that convolution's
        // backward-data, and the other way round
        const cuda::ConvShape cs{(int)B, (int)Cout, (int)Ho, (int)Wo, (int)Cin, K, stride, pad};
        auto out = empty_cuda({B, Cout, Ho, Wo});
        cuda::dnn_conv_backward_data(cs, fp(w), fp(x), wp(out));
        add_bias_c(out, bias, Cout, Ho * Wo);
        attach_fused(out, {x0, w0, b0}, "ConvTranspose2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
            auto G = contiguous_on_device(f32(G0, "ConvTranspose2D"));
            auto dx = empty_cuda({B, Cin, H, W});
            auto dw = empty_cuda({Cin, Cout, (long long)K, (long long)K});
            cuda::dnn_conv_forward(cs, fp(G), fp(w), wp(dx));
            cuda::dnn_conv_backward_filter(cs, fp(G), fp(x), wp(dw));
            return {dx, dw, bias_grad(G, B, Cout, Ho * Wo)};
        });
        return out;
    }
    // col = W^T [CKK, Cin] @ x' [Cin, B*HWi], one gemm, then scattered onto the output
    auto xf = to_cb(x, B, Cin, HWi);
    auto col = empty_cuda({CKK, BHW});
    cuda::sgemm(true, false, (int)CKK, (int)BHW, (int)Cin, 1.0f, static_cast<const float*>(base(w)), (int)CKK,
                static_cast<const float*>(base(xf)), (int)BHW, 0.0f, static_cast<float*>(col->storage()->data()), (int)BHW);
    TensorPtr out;
    {
        NoGrad off;
        out = contiguous(add(col2im_b(col, B, Cout, Ho, Wo, K, stride, pad, H, W), reshape(bias, {1, Cout, 1, 1})));
    }
    attach_fused(out, {x0, w0, b0}, "ConvTranspose2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "ConvTranspose2D"));
        auto gc = im2col_b(G, Cout, Ho, Wo, K, stride, pad, H, W);       // [CKK, B*HWi]
        const float* pgc = static_cast<const float*>(base(gc));
        // dx' = W [Cin, CKK] @ gc -> [Cin, B*HWi]; dW = x' @ gc^T
        auto dxf = empty_cuda({Cin, BHW});
        cuda::sgemm(false, false, (int)Cin, (int)BHW, (int)CKK, 1.0f, static_cast<const float*>(base(w)), (int)CKK, pgc,
                    (int)BHW, 0.0f, static_cast<float*>(dxf->storage()->data()), (int)BHW);
        auto dw = empty_cuda({Cin, Cout, (long long)K, (long long)K});
        cuda::sgemm(false, true, (int)Cin, (int)CKK, (int)BHW, 1.0f, static_cast<const float*>(base(xf)), (int)BHW, pgc,
                    (int)BHW, 0.0f, static_cast<float*>(dw->storage()->data()), (int)CKK);
        auto dx = from_cb(dxf, nullptr, B, Cin, HWi, {B, Cin, H, W});
        auto db = ::sum(::sum(reshape(G, {B, Cout, Ho * Wo}), 2), 0);
        return {dx, dw, db};
    });
    return out;
}

TensorPtr maxpool2d(const TensorPtr& x0, int K, int stride) {
    auto x = contiguous_on_device(f32(x0, "MaxPool2D"));
    long long Bn = x->size(0), C = x->size(1), H = x->size(2), W = x->size(3);
    long long Ho = (H - K) / stride + 1, Wo = (W - K) / stride + 1, planes = Bn * C;
    auto y = empty_cuda({Bn, C, Ho, Wo});
    auto arg = empty_cuda({Bn * C * Ho * Wo}, DType::Int64);
    const void* px = base(x);
    void* py = y->storage()->data();
    void* pa = arg->storage()->data();
    int k = K, s = stride;
    void* args[] = {&px, &py, &pa, &planes, &H, &W, &k, &s, &Ho, &Wo};
    cuda::launch(VK("v_maxpool"), y->numel(), args);
    const Shape oshape{Bn, C, Ho, Wo};
    // op form: a gather of the winners (indices read back only for it)
    Reference reference = [arg, oshape](const vector<TensorPtr>& in) -> TensorPtr {
        auto h = arg->to(Device::CPU);
        auto idx = make_shared<vector<int64_t>>(h->numel());
        memcpy(idx->data(), h->storage()->data(), idx->size() * sizeof(int64_t));
        return gather_flat(in[0], idx, oshape);
    };
    const int64_t nin = x->numel();
    const Shape xshape = x->shape();
    attach_fused(y, {x0}, "MaxPool2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "MaxPool2D"));
        if (stride >= K) {
            auto dx = empty_cuda(xshape);
            const void* pg = base(G);
            const void* pa = arg->storage()->data();
            void* pd = dx->storage()->data();
            long long pls = planes, h = H, w = W, ho = Ho, wo = Wo;
            int k2 = K, s2 = stride;
            void* a[] = {&pg, &pa, &pd, &pls, &h, &w, &k2, &s2, &ho, &wo};
            cuda::launch(VK("v_maxpool_back"), dx->numel(), a);
            return {dx};
        }
        auto dx = make_shared<Tensor>(xshape, F32, Device::CUDA);        // zeroed
        const void* pg = base(G);
        const void* pa = arg->storage()->data();
        void* pd = dx->storage()->data();
        long long n = G->numel();
        void* a[] = {&pg, &pa, &pd, &n};
        cuda::launch(VK("v_scatter_rows"), n, a);
        (void)nin;
        return {dx};
    });
    return y;
}

TensorPtr avgpool2d(const TensorPtr& x0, int K, int stride, Reference reference) {
    auto x = contiguous_on_device(f32(x0, "AvgPool2D"));
    long long Bn = x->size(0), C = x->size(1), H = x->size(2), W = x->size(3);
    long long Ho = (H - K) / stride + 1, Wo = (W - K) / stride + 1, planes = Bn * C;
    auto y = empty_cuda({Bn, C, Ho, Wo});
    const void* px = base(x);
    void* py = y->storage()->data();
    int k = K, s = stride;
    void* args[] = {&px, &py, &planes, &H, &W, &k, &s, &Ho, &Wo};
    cuda::launch(VK("v_avgpool"), y->numel(), args);
    const Shape xshape = x->shape();
    attach_fused(y, {x0}, "AvgPool2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "AvgPool2D"));
        auto dx = make_shared<Tensor>(xshape, F32, Device::CUDA);        // zeroed
        const void* pg = base(G);
        void* pd = dx->storage()->data();
        long long pl = planes, hh = H, ww = W, ho = Ho, wo = Wo;
        int kk = K, ss = stride;
        void* a[] = {&pg, &pd, &pl, &hh, &ww, &kk, &ss, &ho, &wo};
        cuda::launch(VK("v_avgpool_back"), G->numel(), a);
        return {dx};
    });
    return y;
}

// ── BatchNorm and GroupNorm on the device ────────────────────────────────────
//
// BatchNorm: per channel, one block reduces the N*S values (mean, then the
// variance about it) and updates the running estimate in device memory, so a
// training step never waits to read statistics back; a second kernel applies
// the normalisation. The backward is the same shape: one block per channel
// for sum(dy) and sum(dy * xhat), then an element-wise kernel for dx.
// GroupNorm: an (image, group) is one contiguous run of group_size * S values,
// so it is a LayerNorm over that run with a per-channel affine.

static const string& NORM_SRC() {
    static const string s = src(R"(
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w]; __syncthreads(); }
    const float r = sh[0];
    __syncthreads();
    return r;
}
// one block per channel of x viewed as [N, C, S]: mean and 1/std, and the running estimate
extern "C" __global__ void n_bn_stats(const float* x, float* mean, float* inv, float* rm, float* rv,
                                      long long N, long long C, long long S, float eps, float momentum, int update) {
    __shared__ float sh[256];
    const long long c = blockIdx.x, count = N * S;
    float t = 0.0f;
    for (long long i = threadIdx.x; i < count; i += blockDim.x) t += x[((i / S) * C + c) * S + i % S];
    const float m = block_sum(t, sh) / (float)count;
    t = 0.0f;
    for (long long i = threadIdx.x; i < count; i += blockDim.x) { float d = x[((i / S) * C + c) * S + i % S] - m; t += d * d; }
    const float var = block_sum(t, sh) / (float)count;
    if (threadIdx.x == 0) {
        mean[c] = m;
        inv[c] = rsqrtf(var + eps);
        if (update) {
            const float unbiased = count > 1 ? var * (float)count / (float)(count - 1) : var;
            rm[c] = (1.0f - momentum) * rm[c] + momentum * m;
            rv[c] = (1.0f - momentum) * rv[c] + momentum * unbiased;
        }
    }
}
// eval: the statistics are the running estimate
extern "C" __global__ void n_bn_running(const float* rm, const float* rv, float* mean, float* inv, long long C, float eps) {
    long long c = gid(); if (c >= C) return;
    mean[c] = rm[c];
    inv[c] = rsqrtf(rv[c] + eps);
}
extern "C" __global__ void n_affine(const float* x, const float* mean, const float* inv, const float* g, const float* b,
                                    float* y, long long C, long long S, long long n) {
    long long i = gid(); if (i >= n) return;
    const long long c = (i / S) % C;
    y[i] = (x[i] - mean[c]) * inv[c] * g[c] + b[c];
}
// backward sums per channel: sum(dy), sum(dy * xhat) -> also beta's and gamma's gradients
extern "C" __global__ void n_bn_back_sums(const float* x, const float* dy, const float* mean, const float* inv,
                                          float* sdy, float* sdyx, long long N, long long C, long long S) {
    __shared__ float sh[256];
    const long long c = blockIdx.x, count = N * S;
    float a = 0.0f, b = 0.0f;
    for (long long i = threadIdx.x; i < count; i += blockDim.x) {
        const long long j = ((i / S) * C + c) * S + i % S;
        a += dy[j];
        b += dy[j] * (x[j] - mean[c]) * inv[c];
    }
    a = block_sum(a, sh);
    b = block_sum(b, sh);
    if (threadIdx.x == 0) { sdy[c] = a; sdyx[c] = b; }
}
// dx = inv * g * (dy - mean(dy) - xhat * mean(dy * xhat)) in training; inv * g * dy in eval
extern "C" __global__ void n_bn_back_dx(const float* x, const float* dy, const float* mean, const float* inv,
                                        const float* g, const float* sdy, const float* sdyx, float* dx,
                                        long long C, long long S, long long count, long long n, int training) {
    long long i = gid(); if (i >= n) return;
    const long long c = (i / S) % C;
    const float k = inv[c] * g[c];
    if (!training) { dx[i] = k * dy[i]; return; }
    const float xh = (x[i] - mean[c]) * inv[c];
    dx[i] = k * (dy[i] - sdy[c] / (float)count - xh * sdyx[c] / (float)count);
}
// GroupNorm forward: one block per (image, group), a contiguous run of L = gs * S values
extern "C" __global__ void n_gn_forward(const float* x, const float* g, const float* b, float* y, float* mean, float* inv,
                                        long long gs, long long S, long long C, float eps) {
    __shared__ float sh[256];
    const long long L = gs * S, off = (long long)blockIdx.x * L;
    const long long c0 = (blockIdx.x * gs) % C;
    float t = 0.0f;
    for (long long j = threadIdx.x; j < L; j += blockDim.x) t += x[off + j];
    const float m = block_sum(t, sh) / (float)L;
    t = 0.0f;
    for (long long j = threadIdx.x; j < L; j += blockDim.x) { float d = x[off + j] - m; t += d * d; }
    const float iv = rsqrtf(block_sum(t, sh) / (float)L + eps);
    for (long long j = threadIdx.x; j < L; j += blockDim.x) {
        const long long c = c0 + j / S;
        y[off + j] = (x[off + j] - m) * iv * g[c] + b[c];
    }
    if (threadIdx.x == 0) { mean[blockIdx.x] = m; inv[blockIdx.x] = iv; }
}
// GroupNorm backward dx: per (image, group), dx = inv * (g*dy - mean(g*dy) - xhat * mean(g*dy*xhat))
extern "C" __global__ void n_gn_back(const float* x, const float* g, const float* dy, const float* mean, const float* inv,
                                     float* dx, long long gs, long long S, long long C) {
    __shared__ float sh[256];
    const long long L = gs * S, off = (long long)blockIdx.x * L;
    const long long c0 = (blockIdx.x * gs) % C;
    const float m = mean[blockIdx.x], iv = inv[blockIdx.x];
    float a = 0.0f, b = 0.0f;
    for (long long j = threadIdx.x; j < L; j += blockDim.x) {
        const float gd = g[c0 + j / S] * dy[off + j];
        a += gd;
        b += gd * (x[off + j] - m) * iv;
    }
    a = block_sum(a, sh) / (float)L;
    b = block_sum(b, sh) / (float)L;
    for (long long j = threadIdx.x; j < L; j += blockDim.x) {
        const float xh = (x[off + j] - m) * iv;
        dx[off + j] = iv * (g[c0 + j / S] * dy[off + j] - a - xh * b);
    }
}
// GroupNorm gamma / beta: a thread per (image, channel), summed over S, added atomically
extern "C" __global__ void n_gn_back_affine(const float* x, const float* dy, const float* mean, const float* inv,
                                            float* dg, float* db, long long B, long long C, long long gs, long long S) {
    long long i = gid(); if (i >= B * C) return;
    const long long c = i % C, bimg = i / C, grp = bimg * (C / gs) + c / gs;
    const float m = mean[grp], iv = inv[grp];
    const float* xr = x + i * S;
    const float* dr = dy + i * S;
    float ag = 0.0f, ab = 0.0f;
    for (long long s = 0; s < S; s++) { ag += dr[s] * (xr[s] - m) * iv; ab += dr[s]; }
    atomicAdd(&dg[c], ag);
    atomicAdd(&db[c], ab);
}
)");
    return s;
}

static cuda::Kernel NK(const char* name) { return K(name, NORM_SRC()); }

TensorPtr batchnorm(const TensorPtr& x0, const TensorPtr& g0, const TensorPtr& b0, const TensorPtr& rm,
                    const TensorPtr& rv, bool training, float momentum, float eps, Reference reference) {
    auto x = contiguous_on_device(f32(x0, "BatchNorm"));
    auto g = contiguous_on_device(f32(g0, "BatchNorm")), b = contiguous_on_device(f32(b0, "BatchNorm"));
    long long N = x->size(0), C = x->size(1), S = x->numel() / max<long long>(N * C, 1), n = x->numel();
    auto mean = empty_cuda({C}), inv = empty_cuda({C}), y = empty_cuda(x->shape());
    const void* px = base(x);
    void* pm = mean->storage()->data();
    void* pi = inv->storage()->data();
    void* prm = rm->storage()->data();
    void* prv = rv->storage()->data();
    if (training) {
        int upd = 1;
        float mo = momentum, ep = eps;
        void* a[] = {&px, &pm, &pi, &prm, &prv, &N, &C, &S, &ep, &mo, &upd};
        cuda::launch(NK("n_bn_stats"), (unsigned)C, 1, 1, 256, 1, 1, 0, a);
    } else {
        float ep = eps;
        void* a[] = {&prm, &prv, &pm, &pi, &C, &ep};
        cuda::launch(NK("n_bn_running"), C, a);
    }
    const void* pg = base(g);
    const void* pb = base(b);
    void* py = y->storage()->data();
    void* a2[] = {&px, &pm, &pi, &pg, &pb, &py, &C, &S, &n};
    cuda::launch(NK("n_affine"), n, a2);
    attach_fused(y, {x0, g0, b0}, "BatchNorm", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "BatchNorm"));
        auto sdy = empty_cuda({C}), sdyx = empty_cuda({C}), dx = empty_cuda(x->shape());
        const void* px2 = base(x);
        const void* pg2 = base(G);
        const void* pm2 = base(mean);
        const void* pi2 = base(inv);
        void* ps1 = sdy->storage()->data();
        void* ps2 = sdyx->storage()->data();
        long long NN = N, CC = C, SS = S, nn = n, count = N * S;
        void* a3[] = {&px2, &pg2, &pm2, &pi2, &ps1, &ps2, &NN, &CC, &SS};
        cuda::launch(NK("n_bn_back_sums"), (unsigned)C, 1, 1, 256, 1, 1, 0, a3);
        const void* pgam = base(g);
        void* pdx = dx->storage()->data();
        int tr = training ? 1 : 0;
        void* a4[] = {&px2, &pg2, &pm2, &pi2, &pgam, &ps1, &ps2, &pdx, &CC, &SS, &count, &nn, &tr};
        cuda::launch(NK("n_bn_back_dx"), nn, a4);
        return {dx, sdyx, sdy};          // gamma's gradient is sum(dy*xhat), beta's sum(dy)
    });
    return y;
}

TensorPtr groupnorm(const TensorPtr& x0, const TensorPtr& g0, const TensorPtr& b0, int groups, float eps,
                    Reference reference) {
    auto x = contiguous_on_device(f32(x0, "GroupNorm"));
    auto g = contiguous_on_device(f32(g0, "GroupNorm")), b = contiguous_on_device(f32(b0, "GroupNorm"));
    long long B = x->size(0), C = x->size(1), S = x->numel() / max<long long>(B * C, 1), gs = C / groups;
    const long long blocks = B * groups;
    auto y = empty_cuda(x->shape()), mean = empty_cuda({blocks}), inv = empty_cuda({blocks});
    const void* px = base(x);
    const void* pg = base(g);
    const void* pb = base(b);
    void* py = y->storage()->data();
    void* pm = mean->storage()->data();
    void* pi = inv->storage()->data();
    float ep = eps;
    void* a[] = {&px, &pg, &pb, &py, &pm, &pi, &gs, &S, &C, &ep};
    if (blocks) cuda::launch(NK("n_gn_forward"), (unsigned)blocks, 1, 1, 256, 1, 1, 0, a);
    attach_fused(y, {x0, g0, b0}, "GroupNorm", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "GroupNorm"));
        auto dx = empty_cuda(x->shape());
        auto dg = make_shared<Tensor>(Shape{C}, F32, Device::CUDA), db = make_shared<Tensor>(Shape{C}, F32, Device::CUDA);
        const void* px2 = base(x);
        const void* pg2 = base(g);
        const void* pd = base(G);
        const void* pm2 = base(mean);
        const void* pi2 = base(inv);
        void* pdx = dx->storage()->data();
        long long gss = gs, SS = S, CC = C, BB = B;
        void* a1[] = {&px2, &pg2, &pd, &pm2, &pi2, &pdx, &gss, &SS, &CC};
        if (blocks) cuda::launch(NK("n_gn_back"), (unsigned)blocks, 1, 1, 256, 1, 1, 0, a1);
        void* pdg = dg->storage()->data();
        void* pdb = db->storage()->data();
        void* a2[] = {&px2, &pd, &pm2, &pi2, &pdg, &pdb, &BB, &CC, &gss, &SS};
        cuda::launch(NK("n_gn_back_affine"), B * C, a2);
        return {dx, dg, db};
    });
    return y;
}

} // namespace cuda_ops

// ── recurrent layers ─────────────────────────────────────────────────────────
//
// The CPU kernel's scheme on the device: every step's input projection (plus
// bias) in one gemm, then per step one gemm h_{t-1} @ W_h into that step's
// slice and one kernel for the gate arithmetic, which overwrites the slice
// with the activated gates kept for the backward pass. Backward through time
// mirrors it: per step a gate kernel and a gemm back through W_h, then the
// weight gradients over every step at once. Gates are packed side by side,
// [.., G*H]: RNN tanh; LSTM i f o g; GRU r z n.

namespace {

const string& RNN_SRC() {
    static const string s = src(R"(
// sigmoid and tanh on the fast exponential (ex2.approx): the gate kernels
// spent most of their time in five full-precision expf / tanhf per element.
// Within ~1e-7 of the exact functions; tanh saturates cleanly at +-1.
__device__ float sg(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ float th(float x) { return 1.0f - 2.0f / (__expf(2.0f * x) + 1.0f); }
extern "C" __global__ void r_bias_rows(float* y, const float* b, long long rows, int GH) {
    long long i = gid(); if (i >= rows * GH) return;
    y[i] = b[i % GH];
}
// dst [rows, H] = columns [k*H, (k+1)*H) of src [rows, GH]
extern "C" __global__ void r_cols(const float* src, float* dst, long long rows, int H, int GH, int k) {
    long long i = gid(); if (i >= rows * H) return;
    const long long r = i / H; const int j = (int)(i % H);
    dst[i] = src[r * GH + (long long)k * H + j];
}
extern "C" __global__ void r_rnn_fwd(float* a, float* h, int BH) {
    int i = (int)gid(); if (i >= BH) return;
    const float v = th(a[i]); a[i] = v; h[i] = v;
}
extern "C" __global__ void r_lstm_fwd(float* a, const float* c_prev, float* c, float* h, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    float* ar = a + (long long)b * 4 * H;
    const float ig = sg(ar[j]), fg = sg(ar[H + j]), og = sg(ar[2 * H + j]), gg = th(ar[3 * H + j]);
    ar[j] = ig; ar[H + j] = fg; ar[2 * H + j] = og; ar[3 * H + j] = gg;
    const float ct = fg * c_prev[i] + ig * gg;
    c[i] = ct; h[i] = og * th(ct);
}
extern "C" __global__ void r_gru_fwd1(float* a, const float* h_prev, float* rh, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    float* ar = a + (long long)b * 3 * H;
    const float r = sg(ar[j]); ar[j] = r; ar[H + j] = sg(ar[H + j]);
    rh[i] = r * h_prev[i];
}
extern "C" __global__ void r_gru_fwd2(float* a, const float* h_prev, float* h, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    float* ar = a + (long long)b * 3 * H;
    const float n = th(ar[2 * H + j]), z = ar[H + j];
    ar[2 * H + j] = n;
    h[i] = (1.0f - z) * n + z * h_prev[i];
}
// backward: dh holds the gradient arriving from step t+1, dout the output's
extern "C" __global__ void r_rnn_back(const float* g, const float* dout, const float* dh, float* da, int BH) {
    int i = (int)gid(); if (i >= BH) return;
    const float d = dh[i] + dout[i], v = g[i];
    da[i] = d * (1.0f - v * v);
}
extern "C" __global__ void r_lstm_back(const float* g, const float* c, const float* c_prev, const float* dout,
                                       const float* dh, float* dc, float* da, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    const float* gr = g + (long long)b * 4 * H;
    float* dr = da + (long long)b * 4 * H;
    const float ig = gr[j], fg = gr[H + j], og = gr[2 * H + j], gg = gr[3 * H + j];
    const float d = dh[i] + dout[i], tc = th(c[i]);
    const float dct = dc[i] + d * og * (1.0f - tc * tc);
    dr[j]         = dct * gg * ig * (1.0f - ig);
    dr[H + j]     = dct * c_prev[i] * fg * (1.0f - fg);
    dr[2 * H + j] = d * tc * og * (1.0f - og);
    dr[3 * H + j] = dct * ig * (1.0f - gg * gg);
    dc[i] = dct * fg;
}
extern "C" __global__ void r_gru_back1(const float* g, const float* h_prev, const float* dout, const float* dh,
                                       float* da, float* dhp, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    const float* gr = g + (long long)b * 3 * H;
    float* dr = da + (long long)b * 3 * H;
    const float d = dh[i] + dout[i], z = gr[H + j], n = gr[2 * H + j];
    dr[2 * H + j] = d * (1.0f - z) * (1.0f - n * n);
    dr[H + j]     = d * (h_prev[i] - n) * z * (1.0f - z);
    dhp[i] = d * z;
}
extern "C" __global__ void r_gru_back2(const float* g, const float* h_prev, const float* drh, float* da,
                                       float* dhp, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    const float r = g[(long long)b * 3 * H + j];
    da[(long long)b * 3 * H + j] = drh[i] * h_prev[i] * r * (1.0f - r);
    dhp[i] += drh[i] * r;
}
// GRU, PyTorch's form: hh = h W_h for all three gates (no bias), a = the
// input projection plus (b_r, b_z, b_n); a becomes the gates, hn is kept
extern "C" __global__ void r_gruA_fwd(float* a, const float* hh, const float* bhn, const float* h_prev,
                                      float* hn_out, float* h, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    float* ar = a + (long long)b * 3 * H;
    const float* hr = hh + (long long)b * 3 * H;
    const float r = sg(ar[j] + hr[j]), z = sg(ar[H + j] + hr[H + j]);
    const float hn = hr[2 * H + j] + bhn[j];
    const float n = th(ar[2 * H + j] + r * hn);
    ar[j] = r; ar[H + j] = z; ar[2 * H + j] = n;
    hn_out[i] = hn;
    h[i] = (1.0f - z) * n + z * h_prev[i];
}
// its backward: da (input side) and dhh (hidden side, n part dn * r); dh
// becomes d * z, to which the caller adds dhh @ W_h^T
extern "C" __global__ void r_gruA_back(const float* g, const float* h_prev, const float* hn, const float* dout,
                                       float* dh, float* da, float* dhh, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    const long long o = (long long)b * 3 * H;
    const float r = g[o + j], z = g[o + H + j], n = g[o + 2 * H + j];
    const float d = dh[i] + dout[i];
    const float dn = d * (1.0f - z) * (1.0f - n * n);
    const float dzp = d * (h_prev[i] - n) * z * (1.0f - z);
    const float drp = dn * hn[i] * r * (1.0f - r);
    da[o + j] = drp; da[o + H + j] = dzp; da[o + 2 * H + j] = dn;
    dhh[o + j] = drp; dhh[o + H + j] = dzp; dhh[o + 2 * H + j] = dn * r;
    dh[i] = d * z;
}
)");
    return s;
}
cuda::Kernel RK(const char* name) { return K(name, RNN_SRC()); }

const float* cf(const TensorPtr& t) { return static_cast<const float*>(base(t)); }
float* mf(const TensorPtr& t) { return static_cast<float*>(t->storage()->data()); }

// [rows, G*H] -> G tensors [rows, H]
vector<TensorPtr> split_cols(const TensorPtr& src, long long rows, int H, int G) {
    vector<TensorPtr> out;
    for (int k = 0; k < G; k++) {
        auto d = empty_cuda({rows, (long long)H});
        const void* ps = base(src);
        void* pd = d->storage()->data();
        long long rr = rows;
        int h = H, gh = G * H, kk = k;
        void* a[] = {&ps, &pd, &rr, &h, &gh, &kk};
        cuda::launch(RK("r_cols"), rows * H, a);
        out.push_back(d);
    }
    return out;
}

} // namespace

namespace cuda_ops {

TensorPtr rnn(int cell, const TensorPtr& input0, const TensorPtr& h00, const TensorPtr& c00, int H,
              const vector<TensorPtr>& Wx0, const vector<TensorPtr>& Wh0, const vector<TensorPtr>& b0,
              Reference reference) {
    const bool lstm = cell == 1, gru = cell == 2, after = cell == 3;
    auto X = contiguous_on_device(f32(input0, "RNN"));
    const int S = (int)X->size(0), B = (int)X->size(1), I = (int)X->size(2);
    const int G = (int)Wx0.size(), GH = G * H, BH = B * H;
    const long long SB = (long long)S * B;
    TensorPtr h0 = h00 ? contiguous_on_device(f32(h00, "RNN")) : nullptr;
    TensorPtr c0 = (lstm && c00) ? contiguous_on_device(f32(c00, "RNN")) : nullptr;
    TensorPtr Wxp, Whp, WhpT, bp, bhn;
    {
        NoGrad off;
        vector<TensorPtr> wx, wh, bb;
        for (auto& w : Wx0) wx.push_back(f32(w, "RNN"));
        for (auto& w : Wh0) wh.push_back(f32(w, "RNN"));
        for (size_t k = 0; k < (after ? 3 : b0.size()); k++) bb.push_back(f32(b0[k], "RNN"));
        if (after) bhn = contiguous_on_device(f32(b0[3], "RNN"));
        Wxp = contiguous(concat_op(wx, 1));                    // [I, GH]
        Whp = contiguous(concat_op(wh, 1));                    // [H, GH]
        // the backward's products with W_h^T as plain NN gemms against a copy
        // laid out [GH, H]: cuBLAS's NT kernel for 32 x 256 x 256 took 15 us
        // a step against 9 us for NN on a GTX 970
        WhpT = contiguous(transpose(Whp, 0, 1));
        bp  = contiguous(concat_op(bb, 0));                    // [GH]
    }
    auto zeros_bh = make_shared<Tensor>(Shape{(long long)BH}, F32, Device::CUDA);   // zeroed
    const float* h_init = h0 ? cf(h0) : cf(zeros_bh);
    const float* c_init = c0 ? cf(c0) : cf(zeros_bh);

    // 1. every step's input projection plus bias; becomes the activated gates
    auto gates = empty_cuda({SB, (long long)GH});
    {
        const void* pb = base(bp);
        void* pg = gates->storage()->data();
        long long rows = SB;
        int gh = GH;
        void* a[] = {&pg, &pb, &rows, &gh};
        cuda::launch(RK("r_bias_rows"), SB * GH, a);
        cuda::sgemm(false, false, (int)SB, GH, I, 1.0f, cf(X), I, cf(Wxp), GH, 1.0f, mf(gates), GH);
    }
    // outputs: h for every step, then c_n
    auto packed = empty_cuda({SB * H + (lstm ? BH : 0)});
    auto cs = lstm ? empty_cuda({SB, (long long)H}) : nullptr;
    auto rh = gru ? empty_cuda({SB, (long long)H}) : nullptr;
    // reset-after GRU: hn = h W_hn + b_hn per step, and a step's h W_h
    auto HN = after ? empty_cuda({SB, (long long)H}) : nullptr;
    auto hh = after ? empty_cuda({(long long)B, (long long)GH}) : nullptr;
    float* OUT = mf(packed);
    int b_ = B, h_ = H, bh = BH;

    // 2. the recurrence
    for (int t = 0; t < S; t++) {
        const float* hp = t ? OUT + (size_t)(t - 1) * BH : h_init;
        float* at = mf(gates) + (size_t)t * B * GH;
        float* ht = OUT + (size_t)t * BH;
        if (cell == 0) {
            cuda::sgemm(false, false, B, H, H, 1.0f, hp, H, cf(Whp), GH, 1.0f, at, GH);
            void* a[] = {&at, &ht, &bh};
            cuda::launch(RK("r_rnn_fwd"), BH, a);
        } else if (lstm) {
            cuda::sgemm(false, false, B, GH, H, 1.0f, hp, H, cf(Whp), GH, 1.0f, at, GH);
            const float* cp = t ? mf(cs) + (size_t)(t - 1) * BH : c_init;
            float* ct = mf(cs) + (size_t)t * BH;
            void* a[] = {&at, &cp, &ct, &ht, &b_, &h_};
            cuda::launch(RK("r_lstm_fwd"), BH, a);
        } else if (after) {
            // one product for all three gates
            float* phh = mf(hh);
            cuda::sgemm(false, false, B, GH, H, 1.0f, hp, H, cf(Whp), GH, 0.0f, phh, GH);
            const float* pbn = cf(bhn);
            float* hnt = mf(HN) + (size_t)t * BH;
            void* a[] = {&at, &phh, &pbn, &hp, &hnt, &ht, &b_, &h_};
            cuda::launch(RK("r_gruA_fwd"), BH, a);
        } else {
            cuda::sgemm(false, false, B, 2 * H, H, 1.0f, hp, H, cf(Whp), GH, 1.0f, at, GH);
            float* rt = mf(rh) + (size_t)t * BH;
            void* a1[] = {&at, &hp, &rt, &b_, &h_};
            cuda::launch(RK("r_gru_fwd1"), BH, a1);
            cuda::sgemm(false, false, B, H, H, 1.0f, rt, H, cf(Whp) + 2 * H, GH, 1.0f, at + 2 * H, GH);
            void* a2[] = {&at, &hp, &ht, &b_, &h_};
            cuda::launch(RK("r_gru_fwd2"), BH, a2);
        }
    }
    if (lstm) cuda::copy_device(OUT + SB * H, mf(cs) + (size_t)(S - 1) * BH, (size_t)BH * sizeof(float));

    vector<TensorPtr> inputs = {input0};
    if (h00) inputs.push_back(h00);
    if (lstm && c00) inputs.push_back(c00);
    for (auto* group : {&Wx0, &Wh0, &b0}) for (auto& w : *group) inputs.push_back(w);

    attach_fused(packed, inputs, (gru || after) ? "GRU" : lstm ? "LSTM" : "RNN", reference,
                 [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto Gd = contiguous_on_device(f32(G0, "RNN"));
        const float* dOUT = cf(Gd);
        const float* OUTd = cf(packed);
        const float* hi = h0 ? cf(h0) : cf(zeros_bh);
        const float* ci = c0 ? cf(c0) : cf(zeros_bh);
        auto dA = empty_cuda({SB, (long long)GH});
        auto dh = make_shared<Tensor>(Shape{(long long)BH}, F32, Device::CUDA);       // zeroed
        auto dc = lstm ? empty_cuda({(long long)BH}) : nullptr;
        auto dhp = gru ? empty_cuda({(long long)BH}) : nullptr;
        auto drh = gru ? empty_cuda({(long long)BH}) : nullptr;
        auto dHH = after ? empty_cuda({SB, (long long)GH}) : nullptr;
        if (lstm) cuda::copy_device(mf(dc), dOUT + SB * H, (size_t)BH * sizeof(float));
        int b2 = B, h2 = H, bh2 = BH;

        // 3. backward through time
        for (int t = S - 1; t >= 0; t--) {
            const float* hp = t ? OUTd + (size_t)(t - 1) * BH : hi;
            const float* gt = cf(gates) + (size_t)t * B * GH;
            float* da = mf(dA) + (size_t)t * B * GH;
            const float* dout = dOUT + (size_t)t * BH;
            float* pdh = mf(dh);
            if (cell == 0) {
                void* a[] = {&gt, &dout, &pdh, &da, &bh2};
                cuda::launch(RK("r_rnn_back"), BH, a);
                cuda::sgemm(false, false, B, H, GH, 1.0f, da, GH, cf(WhpT), H, 0.0f, pdh, H);
            } else if (lstm) {
                const float* ct = cf(cs) + (size_t)t * BH;
                const float* cp = t ? cf(cs) + (size_t)(t - 1) * BH : ci;
                float* pdc = mf(dc);
                void* a[] = {&gt, &ct, &cp, &dout, &pdh, &pdc, &da, &b2, &h2};
                cuda::launch(RK("r_lstm_back"), BH, a);
                cuda::sgemm(false, false, B, H, GH, 1.0f, da, GH, cf(WhpT), H, 0.0f, pdh, H);
            } else if (after) {
                const float* hnt = cf(HN) + (size_t)t * BH;
                float* dq = mf(dHH) + (size_t)t * B * GH;
                void* a[] = {&gt, &hp, &hnt, &dout, &pdh, &da, &dq, &b2, &h2};
                cuda::launch(RK("r_gruA_back"), BH, a);
                // dh = d * z + dhh @ W_h^T, one product
                cuda::sgemm(false, false, B, H, GH, 1.0f, dq, GH, cf(WhpT), H, 1.0f, pdh, H);
            } else {
                float* pp = mf(dhp);
                float* pr = mf(drh);
                void* a1[] = {&gt, &hp, &dout, &pdh, &da, &pp, &b2, &h2};
                cuda::launch(RK("r_gru_back1"), BH, a1);
                // through (r * h) @ W_hn
                cuda::sgemm(false, false, B, H, H, 1.0f, da + 2 * H, GH, cf(WhpT) + (size_t)2 * H * H, H, 0.0f, pr, H);
                void* a2[] = {&gt, &hp, &pr, &da, &pp, &b2, &h2};
                cuda::launch(RK("r_gru_back2"), BH, a2);
                // through h @ [W_hr W_hz]
                cuda::sgemm(false, false, B, H, 2 * H, 1.0f, da, GH, cf(WhpT), H, 1.0f, pp, H);
                cuda::copy_device(pdh, pp, (size_t)BH * sizeof(float));
            }
        }

        // 4. over every step at once
        vector<TensorPtr> out;
        TensorPtr dX;
        if (input0->requires_grad()) {
            dX = empty_cuda({(long long)S, (long long)B, (long long)I});
            cuda::sgemm(false, true, (int)SB, I, GH, 1.0f, cf(dA), GH, cf(Wxp), GH, 0.0f, mf(dX), I);
        }
        out.push_back(dX);
        if (h00) out.push_back(reshape(dh, {(long long)B, (long long)H}));
        if (lstm && c00) out.push_back(reshape(dc, {(long long)B, (long long)H}));
        auto dWx = empty_cuda({(long long)I, (long long)GH});
        cuda::sgemm(true, false, I, GH, (int)SB, 1.0f, cf(X), I, cf(dA), GH, 0.0f, mf(dWx), GH);
        for (auto& d : split_cols(dWx, I, H, G)) out.push_back(d);
        // h_{t-1} for every step: h0, then the outputs shifted by one
        auto Hprev = empty_cuda({SB, (long long)H});
        cuda::copy_device(mf(Hprev), hi, (size_t)BH * sizeof(float));
        if (S > 1) cuda::copy_device(mf(Hprev) + BH, OUTd, (size_t)(S - 1) * BH * sizeof(float));
        auto dWh = empty_cuda({(long long)H, (long long)GH});
        if (after) {
            cuda::sgemm(true, false, H, GH, (int)SB, 1.0f, cf(Hprev), H, cf(dHH), GH, 0.0f, mf(dWh), GH);
        } else if (!gru) {
            cuda::sgemm(true, false, H, GH, (int)SB, 1.0f, cf(Hprev), H, cf(dA), GH, 0.0f, mf(dWh), GH);
        } else {
            cuda::sgemm(true, false, H, 2 * H, (int)SB, 1.0f, cf(Hprev), H, cf(dA), GH, 0.0f, mf(dWh), GH);
            cuda::sgemm(true, false, H, H, (int)SB, 1.0f, cf(rh), H, cf(dA) + 2 * H, GH, 0.0f, mf(dWh) + 2 * H, GH);
        }
        for (auto& d : split_cols(dWh, H, H, G)) out.push_back(d);
        auto db = ::sum(dA, 0);                                                // [GH]
        for (auto& d : split_cols(db, 1, H, G)) out.push_back(reshape(d, {(long long)H}));
        if (after) {                                                           // b_hn: dhh's n part
            auto dbh = ::sum(dHH, 0);
            out.push_back(reshape(split_cols(dbh, 1, H, G)[2], {(long long)H}));
        }
        return out;
    });
    return packed;
}

} // namespace cuda_ops
