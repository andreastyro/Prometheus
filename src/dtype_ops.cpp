// Dtypes beyond float32: promotion, casts, the float64 / half / integer paths
// of the core ops, comparisons and masks, and autocast. See ops.hpp.
#include "ops_internal.hpp"
#include "ml/strided.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace std;

namespace {

constexpr DType F32 = DType::Float32, F64 = DType::Float64, F16 = DType::Float16,
                BF16 = DType::BFloat16, I64 = DType::Int64, I32 = DType::Int32, BOOL = DType::Bool;

bool is_half(DType d) { return d == F16 || d == BF16; }
bool is_int(DType d)  { return d == I64 || d == I32; }
int  category(DType d) { return d == BOOL ? 0 : is_int(d) ? 1 : 2; }

// Exact integer element access; strided::load goes through double, which
// rounds int64 past 2^53.
int64_t load_i(const void* base, DType d, int64_t i) {
    switch (d) {
        case I64:  return static_cast<const int64_t*>(base)[i];
        case I32:  return static_cast<const int32_t*>(base)[i];
        case BOOL: return static_cast<const bool*>(base)[i] ? 1 : 0;
        default:   return (int64_t)strided::load(base, d, i);
    }
}
void store_i(void* base, DType d, int64_t i, int64_t v) {
    switch (d) {
        case I64:  static_cast<int64_t*>(base)[i] = v; break;
        case I32:  static_cast<int32_t*>(base)[i] = (int32_t)v; break;
        case BOOL: static_cast<bool*>(base)[i] = v != 0; break;
        default:   strided::store(base, d, i, (double)v);
    }
}

// Elements of a contiguous tensor from its storage: index i of the tensor.
struct Reader {
    const void* base; DType d; int64_t off;
    explicit Reader(const TensorPtr& t) : base(t->storage()->data()), d(t->dtype()), off(t->offset()) {}
    double  f(int64_t i) const { return strided::load(base, d, off + i); }
    int64_t n(int64_t i) const { return load_i(base, d, off + i); }
};

// The value side of a contiguous copy, recording no graph.
TensorPtr plain(const TensorPtr& t) {
    if (t->is_contiguous()) return t;
    NoGrad off;
    return contiguous(t);
}

// c * t and c + t with c held at t's precision: a float literal would cost a
// float64 gradient its last eight digits.
TensorPtr cmul(double c, const TensorPtr& t) { return multiply(scalar_tensor(c, t->dtype()), t); }
TensorPtr cadd(double c, const TensorPtr& t) { return add(scalar_tensor(c, t->dtype()), t); }

TensorPtr convert(const TensorPtr& a, DType d) {
    auto out = Tensor::empty(a->shape(), d);
    const void* src = a->storage()->data();
    void* dst = out->storage()->data();
    const DType from = a->dtype();
    if (is_int(from) || from == BOOL)
        strided::for_each(a->shape(), a->strides(), a->offset(), [&](int64_t i, int64_t off) {
            if (is_int(d) || d == BOOL) store_i(dst, d, i, load_i(src, from, off));
            else strided::store(dst, d, i, (double)load_i(src, from, off));
        });
    else
        strided::for_each(a->shape(), a->strides(), a->offset(), [&](int64_t i, int64_t off) {
            strided::store(dst, d, i, strided::load(src, from, off));
        });
    return out;
}

// out[i] = f(a[i]) in double, stored as out_d
TensorPtr map_values(const TensorPtr& a, DType out_d, const function<double(double)>& f) {
    auto A = plain(a);
    auto out = Tensor::empty(A->shape(), out_d);
    const Reader r(A);
    void* dst = out->storage()->data();
    const int64_t n = A->numel();
    const bool int_in = is_int(A->dtype()) || A->dtype() == BOOL;
    if (is_int(out_d) || out_d == BOOL) {
        for (int64_t i = 0; i < n; i++)
            store_i(dst, out_d, i, (int64_t)f(int_in ? (double)r.n(i) : r.f(i)));
    } else {
        #pragma omp parallel for schedule(static) if(n >= 32768)
        for (int64_t i = 0; i < n; i++) strided::store(dst, out_d, i, f(r.f(i)));
    }
    return out;
}

// out = broadcast f(a, b), each element as double
TensorPtr map_binary(const TensorPtr& a, const TensorPtr& b, DType out_d,
                     const function<double(double, double)>& f) {
    auto A = plain(a), B = plain(b);
    const Shape shape = broadcast_shapes(A->shape(), B->shape());
    auto out = Tensor::empty(shape, out_d);
    const Reader ra(A), rb(B);
    void* dst = out->storage()->data();
    broadcast_walk(shape, broadcast_strides(A->shape(), shape), broadcast_strides(B->shape(), shape),
                   false, [&](int64_t i, int64_t ia, int64_t ib) {
                       strided::store(dst, out_d, i, f(ra.f(ia), rb.f(ib))); });
    return out;
}

// the same in exact int64
TensorPtr map_binary_int(const TensorPtr& a, const TensorPtr& b, DType out_d,
                         const function<int64_t(int64_t, int64_t)>& f) {
    auto A = plain(a), B = plain(b);
    const Shape shape = broadcast_shapes(A->shape(), B->shape());
    auto out = Tensor::empty(shape, out_d);
    const Reader ra(A), rb(B);
    void* dst = out->storage()->data();
    broadcast_walk(shape, broadcast_strides(A->shape(), shape), broadcast_strides(B->shape(), shape),
                   false, [&](int64_t i, int64_t ia, int64_t ib) {
                       store_i(dst, out_d, i, f(ra.n(ia), rb.n(ib))); });
    return out;
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

double apply(dispatch::Bin op, double x, double y) {
    switch (op) {
        case dispatch::Bin::Add: return x + y;
        case dispatch::Bin::Sub: return x - y;
        case dispatch::Bin::Mul: return x * y;
        default:                 return x / y;
    }
}

DType g_autocast = DType::Float32;

} // namespace

// ── Promotion and casts ──────────────────────────────────────────────────────

DType promote_types(DType a, DType b) {
    if (a == b) return a;
    const int ca = category(a), cb = category(b);
    if (ca != cb) return ca > cb ? a : b;
    if (ca == 1) return I64;
    if (is_half(a) && is_half(b)) return F32;          // float16 + bfloat16
    auto rank = [](DType d) { return d == F64 ? 3 : d == F32 ? 2 : 1; };
    return rank(a) >= rank(b) ? a : b;
}

DType result_type(const TensorPtr& a, const TensorPtr& b) {
    const bool za = a->dim() == 0, zb = b->dim() == 0;
    if (za == zb) return promote_types(a->dtype(), b->dtype());
    const TensorPtr& t = za ? b : a;     // the dimensioned operand decides,
    const TensorPtr& s = za ? a : b;     // unless the scalar is of a higher category
    return category(s->dtype()) > category(t->dtype()) ? s->dtype() : t->dtype();
}

TensorPtr scalar_tensor(double value, DType dtype) {
    auto t = make_shared<Tensor>(Shape{}, dtype);
    if (is_int(dtype) || dtype == BOOL) store_i(t->storage()->data(), dtype, 0, (int64_t)value);
    else strided::store(t->storage()->data(), dtype, 0, value);
    return t;
}

TensorPtr cast_op(TensorPtr a, DType d) {
    if (!a || a->dtype() == d) return a;
    auto out = a->device() == Device::CUDA ? cuda_ops::cast(a, d) : convert(a, d);
    if (is_floating(d) && is_floating(a->dtype()) && a->requires_grad()) {
        auto node = make_node(out, {a});
        node->name = "cast";
        if (d != F64 && a->dtype() != F64 && a->device() == Device::CPU)
            node->backward_fn = [a](const TensorPtr& out) {
                const float* g = out->grad();
                float* ga = a->grad();
                for (int64_t i = 0; i < out->numel(); i++) ga[i] += g[i];
            };
        // the gradient goes back at the precision a's buffer keeps: double
        // for float64, float32 for everything else -- not rounded to half
        node->backward_graph = [a](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
            return {cast_op(G, a->dtype() == F64 ? F64 : F32)};
        };
    }
    return out;
}

// ── Autocast ─────────────────────────────────────────────────────────────────

DType autocast_dtype() { return g_autocast; }

DType set_autocast_dtype(DType d) {
    if (!is_half(d) && d != F32)
        throw runtime_error(string("autocast: dtype must be float16 or bfloat16, got ") + dtype_name(d));
    const DType previous = g_autocast;
    g_autocast = d;
    return previous;
}

Autocast::Autocast(DType d) : previous(set_autocast_dtype(d)) {}
Autocast::~Autocast() { g_autocast = previous; }

TensorPtr autocast_input(TensorPtr t) {
    if (!t || g_autocast == F32 || !is_floating(t->dtype()) || t->dtype() == F64) return t;
    return cast_op(t, g_autocast);
}

// ── Routing the core ops ─────────────────────────────────────────────────────

namespace dispatch {

TensorPtr unary(const char* name, const TensorPtr& a, function<double(double)> f, Graph1 graph,
                const Self1& self, bool keeps_int, bool range_op, const char* cuda_kernel, float p0, float p1) {
    if (a->device() == Device::CUDA) {
        if (!cuda_kernel) throw runtime_error(string(name) + ": not available on the GPU yet");
        auto r = cuda_ops::unary(cuda_kernel, name, a, p0, p1, graph);
        const bool half = a->dtype() == F16 || a->dtype() == BF16;
        return (half && !(range_op && g_autocast != F32)) ? cast_op(r, a->dtype()) : r;
    }
    const DType d = a->dtype();
    if (d == F32) return nullptr;
    if (is_half(d)) {
        auto r = self(cast_op(a, F32));
        return (range_op && g_autocast != F32) ? r : cast_op(r, d);
    }
    if (d != F64 && !keeps_int) return self(cast_op(a, F32));
    auto out = map_values(a, d, f);
    attach(out, {a}, name, [a, graph](const TensorPtr& out, const TensorPtr& G) -> vector<TensorPtr> {
        return {graph(a, out, G)};
    });
    return out;
}

TensorPtr binary(const char* name, const TensorPtr& a, const TensorPtr& b, Bin op, const Self2& self) {
    if (a->dtype() == F32 && b->dtype() == F32) return nullptr;
    const DType d = result_type(a, b);
    if (d == F32) return self(cast_op(a, F32), cast_op(b, F32));
    if (is_half(d)) return cast_op(self(cast_op(a, F32), cast_op(b, F32)), d);
    if (!is_floating(d)) {
        if (op == Bin::Div) return self(cast_op(a, F32), cast_op(b, F32));   // true division
        if (d == BOOL && op == Bin::Sub)
            throw runtime_error("subtract: not defined for bool tensors; use logical_xor");
        if (d == BOOL)   // + is or, * is and, as in PyTorch
            return map_binary_int(a, b, BOOL, [op](int64_t x, int64_t y) -> int64_t {
                return op == Bin::Add ? (x || y) : (x && y); });
        return map_binary_int(a, b, d, [op](int64_t x, int64_t y) -> int64_t {
            return op == Bin::Add ? x + y : op == Bin::Sub ? x - y : x * y; });
    }
    // float64
    auto A = cast_op(a, F64), B = cast_op(b, F64);
    if (op == Bin::Div) {
        const auto Bc = plain(B);
        const Reader rb(Bc);
        for (int64_t i = 0; i < Bc->numel(); i++)
            if (rb.f(i) == 0.0) throw runtime_error("Division by zero at index " + to_string(i));
    }
    auto out = map_binary(A, B, F64, [op](double x, double y) { return apply(op, x, y); });
    attach(out, {A, B}, name, [A, B, op](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        TensorPtr ga, gb;
        switch (op) {
            case Bin::Add: ga = G; gb = G; break;
            case Bin::Sub: ga = G; gb = cmul(-1.0, G); break;
            case Bin::Mul: ga = multiply(G, B); gb = multiply(G, A); break;
            case Bin::Div: ga = divide(G, B); gb = cmul(-1.0, divide(multiply(G, A), multiply(B, B))); break;
        }
        return {A->requires_grad() ? sum_to(ga, A->shape()) : nullptr,
                B->requires_grad() ? sum_to(gb, B->shape()) : nullptr};
    });
    return out;
}

TensorPtr reduce(const char* name, const TensorPtr& a, int dim, bool keepdim, bool mean,
                 const Self1& self) {
    const DType d = a->dtype();
    if (d == F32) return nullptr;
    if (is_half(d)) {
        auto r = self(cast_op(a, F32));
        return g_autocast != F32 ? r : cast_op(r, d);
    }
    if (d != F64 && mean) return self(cast_op(a, F32));

    const auto A = plain(a);
    const int nd = A->dim();
    int64_t outer = 1, len = A->numel(), inner = 1;
    Shape out_shape;
    if (dim == -1) {
        out_shape = keepdim ? Shape(max(nd, 1), 1) : Shape{1};
    } else {
        if (dim < 0 || dim >= nd)
            throw runtime_error(string(name) + ": dim " + to_string(dim) + " is out of range for a " +
                                to_string(nd) + "-D tensor (-1 reduces everything)");
        outer = 1; inner = 1;
        for (int k = 0; k < dim; k++) outer *= A->shape()[k];
        for (int k = dim + 1; k < nd; k++) inner *= A->shape()[k];
        len = A->shape()[dim];
        for (int k = 0; k < nd; k++)
            if (k != dim) out_shape.push_back(A->shape()[k]);
            else if (keepdim) out_shape.push_back(1);
        if (out_shape.empty()) out_shape = {1};
    }
    const Reader r(A);
    if (d != F64) {          // integer or bool: an exact int64 sum, as PyTorch
        auto out = Tensor::zeros(out_shape, I64);
        int64_t* o = out->data_ptr<int64_t>();
        for (int64_t p = 0; p < outer; p++)
            for (int64_t k = 0; k < len; k++)
                for (int64_t j = 0; j < inner; j++) o[p * inner + j] += r.n((p * len + k) * inner + j);
        return out;
    }
    auto out = Tensor::zeros(out_shape, F64);
    double* o = out->data_ptr<double>();
    for (int64_t p = 0; p < outer; p++)
        for (int64_t k = 0; k < len; k++)
            for (int64_t j = 0; j < inner; j++) o[p * inner + j] += r.f((p * len + k) * inner + j);
    if (mean) for (int64_t i = 0; i < out->numel(); i++) o[i] /= (double)len;

    attach(out, {A}, name, [A, dim, len, mean](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        TensorPtr g;
        if (dim == -1) {
            g = expand_to(reshape(G, {1}), A->shape());
        } else {
            Shape kept = A->shape();
            kept[dim] = 1;
            g = expand_to(reshape(G, kept), A->shape());
        }
        return {mean ? cmul(1.0 / (double)len, g) : g};
    });
    return out;
}

TensorPtr extreme(const char* name, const TensorPtr& a, bool is_max, const Self1&) {
    const DType d = a->dtype();
    if (d == F32) return nullptr;
    const auto A = plain(a);
    if (A->numel() == 0) throw runtime_error(string(name) + ": empty tensor");
    const Reader r(A);
    int64_t arg = 0;
    if (is_int(d) || d == BOOL) {
        for (int64_t i = 1; i < A->numel(); i++)
            if (is_max ? r.n(i) > r.n(arg) : r.n(i) < r.n(arg)) arg = i;
    } else {
        for (int64_t i = 1; i < A->numel(); i++)
            if (is_max ? r.f(i) > r.f(arg) : r.f(i) < r.f(arg)) arg = i;
    }
    auto idx = make_shared<const vector<int64_t>>(1, arg);
    TensorPtr out;
    {
        NoGrad off;
        out = gather_flat(A, idx, Shape{1});
    }
    attach(out, {A}, name, [A, idx](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        return {scatter_add_flat(reshape(G, {1}), idx, A->shape())};
    });
    return out;
}

TensorPtr argmax(const TensorPtr& a) {
    if (a->dtype() == F32) return nullptr;
    const auto A = plain(a);
    if (A->dim() < 1) throw runtime_error("argmax: needs at least one dimension");
    const int64_t cols = A->shape().back();
    const int64_t rows = cols ? A->numel() / cols : 0;
    Shape out_shape(A->shape().begin(), A->shape().end() - 1);
    if (out_shape.empty()) out_shape = {1};
    auto out = make_shared<Tensor>(out_shape);
    const Reader r(A);
    const bool integral = is_int(A->dtype()) || A->dtype() == BOOL;
    for (int64_t i = 0; i < rows; i++) {
        int64_t best = 0;
        for (int64_t j = 1; j < cols; j++)
            if (integral ? r.n(i * cols + j) > r.n(i * cols + best) : r.f(i * cols + j) > r.f(i * cols + best))
                best = j;
        out->data()[i] = (float)best;
    }
    return out;
}

TensorPtr softmax(const TensorPtr& a, const Self1& self) {
    const DType d = a->dtype();
    if (d == F32) return nullptr;
    if (is_half(d)) {
        auto r = self(cast_op(a, F32));
        return g_autocast != F32 ? r : cast_op(r, d);
    }
    if (d != F64) return self(cast_op(a, F32));
    const auto A = plain(a);
    if (A->dim() < 1) throw runtime_error("softmax: needs at least one dimension");
    const int64_t cols = A->shape().back();
    const int64_t rows = cols ? A->numel() / cols : 0;
    auto out = Tensor::empty(A->shape(), F64);
    const double* x0 = A->data_ptr<double>();
    double* y0 = out->data_ptr<double>();
    #pragma omp parallel for schedule(static) if(rows * cols >= 32768)
    for (int64_t i = 0; i < rows; i++) {
        const double* x = x0 + i * cols;
        double* y = y0 + i * cols;
        double mx = x[0], total = 0.0;
        for (int64_t j = 1; j < cols; j++) mx = max(mx, x[j]);
        for (int64_t j = 0; j < cols; j++) { y[j] = std::exp(x[j] - mx); total += y[j]; }
        for (int64_t j = 0; j < cols; j++) y[j] /= total;
    }
    attach(out, {A}, "softmax", [](const TensorPtr& out, const TensorPtr& G) -> vector<TensorPtr> {
        auto s = sum(multiply(G, out), out->dim() - 1, true);
        return {multiply(out, subtract(G, s))};
    });
    return out;
}

TensorPtr matmul(TensorPtr a, TensorPtr b, const Self2& self) {
    if (a->dtype() == F32 && b->dtype() == F32) return nullptr;
    const DType d = result_type(a, b);
    if (d == F32 || is_half(d)) {
        // the float32 kernel, with autocast off so it does not cast straight back
        Autocast off(F32);
        auto r = self(cast_op(a, F32), cast_op(b, F32));
        return d == F32 ? r : cast_op(r, d);
    }

    // float64, and integers through float64 (exact while products stay under 2^53)
    auto A = cast_op(a, F64), B = cast_op(b, F64);
    A = A->is_contiguous() ? A : contiguous(A);
    B = B->is_contiguous() ? B : contiguous(B);
    const int64_t m = A->size(-2), k = A->size(-1), n = B->size(-1);
    const Shape batch_a(A->shape().begin(), A->shape().end() - 2);
    const Shape batch_b(B->shape().begin(), B->shape().end() - 2);
    const Shape batch = broadcast_shapes(batch_a, batch_b);
    Shape out_shape = batch;
    out_shape.push_back(m);
    out_shape.push_back(n);
    auto out = Tensor::zeros(out_shape, F64);

    int64_t nb = 1;
    for (int64_t v : batch) nb *= v;
    vector<int64_t> ia((size_t)nb, 0), ib((size_t)nb, 0);
    if (!batch.empty())
        broadcast_walk(batch, broadcast_strides(batch_a, batch), broadcast_strides(batch_b, batch), false,
                       [&](int64_t i, int64_t x, int64_t y) { ia[i] = x; ib[i] = y; });
    const double* pa = A->data_ptr<double>();
    const double* pb = B->data_ptr<double>();
    double* pc = out->data_ptr<double>();
    for (int64_t q = 0; q < nb; q++) {
        const double* Am = pa + ia[q] * m * k;
        const double* Bm = pb + ib[q] * k * n;
        double* Cm = pc + q * m * n;
        #pragma omp parallel for schedule(static) if(m * n * k >= (1 << 16))
        for (int64_t i = 0; i < m; i++)
            for (int64_t p = 0; p < k; p++) {
                const double aip = Am[i * k + p];
                const double* brow = Bm + p * n;
                double* crow = Cm + i * n;
                for (int64_t j = 0; j < n; j++) crow[j] += aip * brow[j];
            }
    }
    attach(out, {A, B}, "matmul", [A, B](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        return {A->requires_grad() ? sum_to(::matmul(G, transpose(B)), A->shape()) : nullptr,
                B->requires_grad() ? sum_to(::matmul(transpose(A), G), B->shape()) : nullptr};
    });
    return is_floating(d) ? out : cast_op(out, d);
}

TensorPtr concat(vector<TensorPtr>& inputs, int axis, const function<TensorPtr(vector<TensorPtr>, int)>& self) {
    bool all32 = true;
    for (auto& t : inputs) all32 = all32 && t && t->dtype() == F32;
    if (all32 || inputs.empty()) return nullptr;
    DType d = inputs[0]->dtype();
    for (auto& t : inputs) d = promote_types(d, t->dtype());
    if (d == F32 || is_half(d)) {
        vector<TensorPtr> c;
        for (auto& t : inputs) c.push_back(cast_op(t, F32));
        auto r = self(c, axis);
        return d == F32 ? r : cast_op(r, d);
    }
    vector<TensorPtr> c;
    for (auto& t : inputs) {
        auto x = cast_op(t, d);
        c.push_back(x->is_contiguous() ? x : contiguous(x));
    }
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
    auto out = Tensor::empty(out_shape, d);
    int64_t outer = 1, inner = 1;
    for (int k = 0; k < axis; k++) outer *= shape0[k];
    for (int k = axis + 1; k < ndim; k++) inner *= shape0[k];
    const size_t es = dtype_size(d);
    unsigned char* dst = static_cast<unsigned char*>(out->storage()->data());
    for (int64_t o = 0; o < outer; o++) {
        int64_t written = 0;
        for (auto& t : c) {
            const int64_t span = t->shape()[axis] * inner;
            const unsigned char* src = static_cast<const unsigned char*>(t->storage()->data()) +
                                       (size_t)(t->offset() + o * span) * es;
            memcpy(dst + (size_t)(o * total * inner + written) * es, src, (size_t)span * es);
            written += span;
        }
    }
    attach(out, c, "concat_op", [c, outer, inner, total, axis](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        vector<TensorPtr> parts;
        int64_t offset = 0;
        for (auto& t : c) {
            const int64_t len = t->shape()[axis];
            if (t->requires_grad()) {
                auto idx = make_shared<vector<int64_t>>();
                for (int64_t o = 0; o < outer; o++)
                    for (int64_t j = 0; j < len; j++)
                        for (int64_t r = 0; r < inner; r++)
                            idx->push_back((o * total + offset + j) * inner + r);
                parts.push_back(gather_flat(G, idx, t->shape()));
            } else {
                parts.push_back(nullptr);
            }
            offset += len;
        }
        return parts;
    });
    return out;
}

TensorPtr gather(const TensorPtr& x, shared_ptr<const vector<int64_t>> idx, const Shape& out_shape) {
    const DType d = x->dtype();
    if (d == F32) return nullptr;
    if (is_half(d)) return cast_op(gather_flat(cast_op(x, F32), idx, out_shape), d);
    const auto X = plain(x);
    auto out = Tensor::empty(out_shape, d);
    if ((int64_t)idx->size() != out->numel())
        throw runtime_error("gather_flat: " + to_string(idx->size()) + " indices for " +
                            to_string(out->numel()) + " outputs");
    const size_t es = dtype_size(d);
    const unsigned char* src = static_cast<const unsigned char*>(X->storage()->data()) + (size_t)X->offset() * es;
    unsigned char* dst = static_cast<unsigned char*>(out->storage()->data());
    for (int64_t i = 0; i < out->numel(); i++) {
        const int64_t j = (*idx)[(size_t)i];
        if (j >= X->numel()) throw runtime_error("gather_flat: index " + to_string(j) + " out of range");
        if (j < 0) memset(dst + i * es, 0, es);
        else       memcpy(dst + i * es, src + j * es, es);
    }
    attach(out, {X}, "gather", [X, idx](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        return {scatter_add_flat(G, idx, X->shape())};
    });
    return out;
}

TensorPtr scatter_add(const TensorPtr& g, shared_ptr<const vector<int64_t>> idx, const Shape& out_shape) {
    const DType d = g->dtype();
    if (d == F32) return nullptr;
    if (is_half(d)) return cast_op(scatter_add_flat(cast_op(g, F32), idx, out_shape), d);
    const auto Gt = plain(g);
    if ((int64_t)idx->size() != Gt->numel())
        throw runtime_error("scatter_add_flat: " + to_string(idx->size()) + " indices for " +
                            to_string(Gt->numel()) + " values");
    auto out = Tensor::zeros(out_shape, d);
    const Reader r(Gt);
    void* dst = out->storage()->data();
    for (int64_t i = 0; i < Gt->numel(); i++) {
        const int64_t j = (*idx)[(size_t)i];
        if (j >= out->numel()) throw runtime_error("scatter_add_flat: index " + to_string(j) + " out of range");
        if (j < 0) continue;
        if (d == F64) out->data_ptr<double>()[j] += r.f(i);
        else          store_i(dst, d, j, load_i(dst, d, j) + r.n(i));
    }
    attach(out, {Gt}, "scatter_add", [Gt, idx](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        return {gather_flat(G, idx, Gt->shape())};
    });
    return out;
}

} // namespace dispatch

// ── Comparisons, logic, masks ────────────────────────────────────────────────

namespace {

enum class Cmp { EQ, NE, LT, LE, GT, GE };

template <class T> bool cmp(Cmp c, T x, T y) {
    switch (c) {
        case Cmp::EQ: return x == y;
        case Cmp::NE: return x != y;
        case Cmp::LT: return x < y;
        case Cmp::LE: return x <= y;
        case Cmp::GT: return x > y;
        default:      return x >= y;
    }
}

TensorPtr compare(const TensorPtr& a, const TensorPtr& b, Cmp c) {
    if (a->device() == Device::CUDA || b->device() == Device::CUDA)
        return cuda_ops::compare((int)c, a, b);
    const bool integral = !is_floating(a->dtype()) && !is_floating(b->dtype());
    if (integral)
        return map_binary_int(a, b, BOOL, [c](int64_t x, int64_t y) -> int64_t { return cmp(c, x, y); });
    return map_binary(a, b, BOOL, [c](double x, double y) -> double { return cmp(c, x, y); });
}

TensorPtr logic(const TensorPtr& a, const TensorPtr& b, int op) {
    if (a->device() == Device::CUDA || b->device() == Device::CUDA)
        return cuda_ops::compare(6 + op, a, b);
    return map_binary(a, b, BOOL, [op](double x, double y) -> double {
        const bool p = x != 0.0, q = y != 0.0;
        return op == 0 ? (p && q) : op == 1 ? (p || q) : (p != q);
    });
}

} // namespace

TensorPtr eq(TensorPtr a, TensorPtr b) { return compare(a, b, Cmp::EQ); }
TensorPtr ne(TensorPtr a, TensorPtr b) { return compare(a, b, Cmp::NE); }
TensorPtr lt(TensorPtr a, TensorPtr b) { return compare(a, b, Cmp::LT); }
TensorPtr le(TensorPtr a, TensorPtr b) { return compare(a, b, Cmp::LE); }
TensorPtr gt(TensorPtr a, TensorPtr b) { return compare(a, b, Cmp::GT); }
TensorPtr ge(TensorPtr a, TensorPtr b) { return compare(a, b, Cmp::GE); }
TensorPtr eq(TensorPtr a, double s) { return compare(a, scalar_tensor(s, F64), Cmp::EQ); }
TensorPtr ne(TensorPtr a, double s) { return compare(a, scalar_tensor(s, F64), Cmp::NE); }
TensorPtr lt(TensorPtr a, double s) { return compare(a, scalar_tensor(s, F64), Cmp::LT); }
TensorPtr le(TensorPtr a, double s) { return compare(a, scalar_tensor(s, F64), Cmp::LE); }
TensorPtr gt(TensorPtr a, double s) { return compare(a, scalar_tensor(s, F64), Cmp::GT); }
TensorPtr ge(TensorPtr a, double s) { return compare(a, scalar_tensor(s, F64), Cmp::GE); }

TensorPtr logical_and(TensorPtr a, TensorPtr b) { return logic(a, b, 0); }
TensorPtr logical_or(TensorPtr a, TensorPtr b)  { return logic(a, b, 1); }
TensorPtr logical_xor(TensorPtr a, TensorPtr b) { return logic(a, b, 2); }
TensorPtr logical_not(TensorPtr a) {
    if (a->device() == Device::CUDA) return cuda_ops::compare(0, a, scalar_tensor(0.0, F32));
    return map_values(a, BOOL, [](double x) -> double { return x == 0.0; });
}

TensorPtr any(TensorPtr a) {
    const auto A = plain(a);
    const Reader r(A);
    bool v = false;
    for (int64_t i = 0; i < A->numel() && !v; i++) v = r.f(i) != 0.0;
    return Tensor::full({1}, v ? 1.0f : 0.0f, BOOL);
}
TensorPtr all(TensorPtr a) {
    const auto A = plain(a);
    const Reader r(A);
    bool v = true;
    for (int64_t i = 0; i < A->numel() && v; i++) v = r.f(i) != 0.0;
    return Tensor::full({1}, v ? 1.0f : 0.0f, BOOL);
}

TensorPtr isnan_op(TensorPtr a)    { return map_values(a, BOOL, [](double x) -> double { return std::isnan(x); }); }
TensorPtr isinf_op(TensorPtr a)    { return map_values(a, BOOL, [](double x) -> double { return std::isinf(x); }); }
TensorPtr isfinite_op(TensorPtr a) { return map_values(a, BOOL, [](double x) -> double { return std::isfinite(x); }); }

TensorPtr where(TensorPtr cond, TensorPtr a, TensorPtr b) {
    if (cond->device() == Device::CUDA || a->device() == Device::CUDA || b->device() == Device::CUDA) {
        const DType d = result_type(a, b);
        auto r = cuda_ops::where(cond, a, b);
        return (d == F16 || d == BF16) ? cast_op(r, d) : r;
    }
    const DType d = result_type(a, b);
    auto A = cast_op(a, d), B = cast_op(b, d);
    const auto C = plain(cond), Ap = plain(A), Bp = plain(B);
    const Shape shape = broadcast_shapes(broadcast_shapes(C->shape(), Ap->shape()), Bp->shape());
    auto out = Tensor::empty(shape, d);
    const Shape sc = broadcast_strides(C->shape(), shape), sa = broadcast_strides(Ap->shape(), shape),
                sb = broadcast_strides(Bp->shape(), shape);
    const Reader rc(C);
    const size_t es = dtype_size(d);
    const unsigned char* pa = static_cast<const unsigned char*>(Ap->storage()->data()) + (size_t)Ap->offset() * es;
    const unsigned char* pb = static_cast<const unsigned char*>(Bp->storage()->data()) + (size_t)Bp->offset() * es;
    unsigned char* po = static_cast<unsigned char*>(out->storage()->data());
    const int nd = (int)shape.size();
    vector<int64_t> pos(nd, 0);
    int64_t ic = 0, ia = 0, ib = 0;
    for (int64_t i = 0; i < out->numel(); i++) {
        const bool take_a = rc.f(ic) != 0.0;
        memcpy(po + i * es, take_a ? pa + ia * es : pb + ib * es, es);
        for (int k = nd - 1; k >= 0; k--) {     // advance the multi-index
            if (++pos[k] < shape[k]) { ic += sc[k]; ia += sa[k]; ib += sb[k]; break; }
            ic -= sc[k] * (shape[k] - 1); ia -= sa[k] * (shape[k] - 1); ib -= sb[k] * (shape[k] - 1);
            pos[k] = 0;
        }
    }
    attach(out, {A, B}, "where", [C, A, B](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        auto zero = scalar_tensor(0.0, G->dtype());
        return {A->requires_grad() ? sum_to(where(C, G, zero), A->shape()) : nullptr,
                B->requires_grad() ? sum_to(where(C, zero, G), B->shape()) : nullptr};
    });
    return out;
}

TensorPtr masked_fill(TensorPtr a, TensorPtr mask, double value) {
    if (broadcast_shapes(mask->shape(), a->shape()) != a->shape())
        throw runtime_error("masked_fill: mask " + shape_str(mask->shape()) +
                            " does not broadcast to " + shape_str(a->shape()));
    return where(mask, scalar_tensor(value, a->dtype()), a);
}

static TensorPtr floor_op(TensorPtr a, TensorPtr b, bool rem) {
    const char* name = rem ? "remainder" : "floor_divide";
    DType d = result_type(a, b);
    if (!is_floating(d)) {
        if (d == BOOL) d = I64;
        {   // checked before the loop: an exception cannot leave a parallel region
            const auto Bc = plain(b);
            const Reader rb(Bc);
            for (int64_t i = 0; i < Bc->numel(); i++)
                if (rb.n(i) == 0) throw runtime_error(string(name) + ": integer division by zero");
        }
        return map_binary_int(a, b, d, [rem](int64_t x, int64_t y) -> int64_t {
            int64_t q = x / y;
            if ((x % y != 0) && ((x < 0) != (y < 0))) q--;      // floor, not truncate
            return rem ? x - q * y : q;
        });
    }
    return map_binary(a, b, d, [rem](double x, double y) {
        const double q = std::floor(x / y);
        return rem ? x - q * y : q;
    });
}
TensorPtr floor_divide(TensorPtr a, TensorPtr b) { return floor_op(a, b, false); }
TensorPtr remainder(TensorPtr a, TensorPtr b)    { return floor_op(a, b, true); }
