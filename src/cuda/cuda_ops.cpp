// Ops on GPU tensors: kernels compiled at runtime by NVRTC. See ml/cuda.hpp.
//
// Every GPU op computes its forward with a kernel and attaches a
// backward_graph -- its gradient written with ops, which run on the GPU too.
// The autograd engine runs that graph for any node on the GPU, and the
// gradients accumulate on the device. There are no raw backward kernels to
// keep in step with the forward, and the gradients are differentiable again.
#include "../ops_internal.hpp"
#include "ml/cuda.hpp"
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
static TensorPtr f32(const TensorPtr& t, const char* op) {
    auto d = on_device(t, op);
    if (d->dtype() == F32) return d;
    if (d->dtype() == DType::Float64 || d->dtype() == DType::Float16 || d->dtype() == DType::BFloat16 ||
        d->dtype() == DType::Int64 || d->dtype() == DType::Int32 || d->dtype() == DType::Bool)
        return cast(d, F32);
    return d;
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
    attach(out, {a}, name, [a, graph](const TensorPtr& out, const TensorPtr& G) -> vector<TensorPtr> {
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
