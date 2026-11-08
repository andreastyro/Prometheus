// Ops on GPU tensors: kernels compiled at runtime by NVRTC. See ml/cuda.hpp.
//
// Every GPU op computes its forward with a kernel and attaches a
// backward_graph -- its gradient written with ops, which run on the GPU too.
// The autograd engine runs that graph for any node on the GPU, and the
// gradients accumulate on the device. There are no raw backward kernels to
// keep in step with the forward, and the gradients are differentiable again.
#include "../ops_internal.hpp"
#include "ml/cuda.hpp"
#include <cstring>
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
// element i of the output, read through a strided / broadcast operand
__device__ long long at(const Index& ix, long long i) {
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
    return ix;
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
    if (da >= 4 && dy >= 4) sti(y, dy, i, ldi(a, da, o));
    else st(y, dy, i, ld(a, da, o));
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
    attach(out, {a}, name, [a, graph, is_gelu](const TensorPtr& out, const TensorPtr& G) -> vector<TensorPtr> {
        if (is_gelu && !grad_enabled()) return {gelu_backward(a, G)};
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
    auto a = on_device(a0, "compare"), b = on_device(b0, "compare");
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
    for (int b = 0; b < es; b++) dst[(o * row + offset + j) * es + b] = src[i * es + b];
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
        for (auto& t : c) {
            const int64_t len = t->shape()[axis];
            if (t->requires_grad()) {
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
// LayerNorm, one block per row: dx, and the row's mean and 1/std for the
// gamma / beta column sums.  dx = inv * (g*dy - mean(g*dy) - xhat * mean(g*dy*xhat))
extern "C" __global__ void b_layernorm_rows(const float* x, const float* g, const float* dy, float* dx,
                                            float* mean_out, float* inv_out, long long cols, float eps) {
    __shared__ float s[256];
    __shared__ float s2[256];
    const float* xr = x + (long long)blockIdx.x * cols;
    const float* dyr = dy + (long long)blockIdx.x * cols;
    float* dxr = dx + (long long)blockIdx.x * cols;
    float t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) t += xr[j];
    s[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w]; __syncthreads(); }
    const float mean = s[0] / (float)cols;
    __syncthreads();
    t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) { float d = xr[j] - mean; t += d * d; }
    s[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w]; __syncthreads(); }
    const float inv = rsqrtf(s[0] / (float)cols + eps);
    __syncthreads();
    float a = 0.0f, b = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) {
        const float gd = g[j] * dyr[j];
        a += gd;
        b += gd * (xr[j] - mean) * inv;
    }
    s[threadIdx.x] = a;
    s2[threadIdx.x] = b;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) { s[threadIdx.x] += s[threadIdx.x + w]; s2[threadIdx.x] += s2[threadIdx.x + w]; }
        __syncthreads();
    }
    const float m1 = s[0] / (float)cols, m2 = s2[0] / (float)cols;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) {
        const float xh = (xr[j] - mean) * inv;
        dxr[j] = inv * (g[j] * dyr[j] - m1 - xh * m2);
    }
    if (threadIdx.x == 0) { mean_out[blockIdx.x] = mean; inv_out[blockIdx.x] = inv; }
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
// embedding: rows of the table by id, and the scatter-add back (ids repeat)
extern "C" __global__ void b_embed_fwd(const float* w, const long long* ids, float* out, long long rows, long long e) {
    long long i = gid(); if (i >= rows * e) return;
    out[i] = w[ids[i / e] * e + i % e];
}
extern "C" __global__ void b_embed_bwd(const float* g, const long long* ids, float* dw, long long rows, long long e) {
    long long i = gid(); if (i >= rows * e) return;
    atomicAdd(&dw[ids[i / e] * e + i % e], g[i]);
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
    void* a1[] = {&px, &pg, &pd, &pdx, &pm, &pi, &cols, &eps};
    if (rows) cuda::launch(K("b_layernorm_rows", BACKWARD_SRC()), (unsigned)rows, 1, 1, 256, 1, 1, 0, a1);
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

TensorPtr embedding(const TensorPtr& W0, const vector<int64_t>& ids, const Shape& out_shape) {
    auto W = contiguous_on_device(f32(W0, "Embedding"));
    long long e = W->shape().back(), rows = (long long)ids.size();
    auto host = make_shared<Tensor>(Shape{rows}, DType::Int64);
    if (rows) memcpy(host->storage()->data(), ids.data(), ids.size() * sizeof(int64_t));
    auto dev_ids = host->to(Device::CUDA);
    auto out = empty_cuda(out_shape);
    const void* pw = base(W);
    const void* pi = dev_ids->storage()->data();
    void* po = out->storage()->data();
    void* args[] = {&pw, &pi, &po, &rows, &e};
    cuda::launch(K("b_embed_fwd", BACKWARD_SRC()), rows * e, args);
    if (W->requires_grad()) {
        auto node = make_node(out, {W});
        node->name = "Embedding";
        // the op form, for gradients that are themselves differentiated
        autograd::attach_reference(*node, [ids, e, out_shape](const vector<TensorPtr>& in) -> TensorPtr {
            auto idx = make_shared<vector<int64_t>>(ids.size() * (size_t)e);
            for (size_t r = 0; r < ids.size(); r++)
                for (long long d = 0; d < e; d++) (*idx)[r * e + d] = ids[r] * e + d;
            return gather_flat(in[0], idx, out_shape);
        });
        auto reference = node->backward_graph;
        node->backward_graph = [reference, dev_ids, rows, e, W](const TensorPtr& out, const TensorPtr& G0) -> vector<TensorPtr> {
            if (grad_enabled()) return reference(out, G0);
            auto G = contiguous_on_device(f32(G0, "Embedding"));
            auto dw = make_shared<Tensor>(W->shape(), F32, Device::CUDA);      // zeroed
            const void* pg = base(G);
            const void* pi = dev_ids->storage()->data();
            void* pd = dw->storage()->data();
            long long r = rows, ee = e;
            void* a[] = {&pg, &pi, &pd, &r, &ee};
            cuda::launch(K("b_embed_bwd", BACKWARD_SRC()), r * ee, a);
            return {dw};
        };
    }
    return out;
}

} // namespace cuda_ops
