#include "ml/tensor.hpp"
#include "ml/autograd.hpp"
#include "ml/ops.hpp"
#include "ml/matmul_backend.hpp"
#include "ml/fastmath.hpp"
#include "ml/strided.hpp"
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <string>

using namespace std;

// Every op below reads its inputs through data(), the raw row-major pointer.
// checked_input() makes that safe: a non-contiguous view is materialised
// (the copy joins the graph, so gradients still reach the view's source), and
// a dtype or device the kernels cannot read raises instead of being
// reinterpreted as floats.
TensorPtr checked_input(const char* op, TensorPtr t) {
    if (!t) return t;
    if (t->device() != Device::CPU)
        throw runtime_error(string(op) + ": tensor is on " + device_name(t->device()) +
                            "; this build computes on cpu only");
    if (t->dtype() != DType::Float32)
        throw runtime_error(string(op) + ": expects float32, got " + dtype_name(t->dtype()) +
                            "; convert with .to(float32)");
    return t->is_contiguous() ? t : contiguous(t);
}

// ── Broadcasting ─────────────────────────────────────────────────────────────
//
// Two shapes broadcast when, aligned from the right, each pair of sizes is
// equal or one of them is 1; a missing leading dimension counts as 1. The
// result takes the larger size in every position. This is numpy's rule and
// PyTorch's, so [rows, n] + [n] adds a bias to every row and [b, 1, n] *
// [m, 1] gives [b, m, n].

static string shape_str(const Shape& s) {
    string r = "[";
    for (size_t i = 0; i < s.size(); i++) r += (i ? ", " : "") + to_string(s[i]);
    return r + "]";
}

Shape broadcast_shapes(const Shape& a, const Shape& b) {
    const size_t nd = max(a.size(), b.size());
    Shape out(nd);
    for (size_t i = 0; i < nd; i++) {
        const int64_t da = i < nd - a.size() ? 1 : a[i - (nd - a.size())];
        const int64_t db = i < nd - b.size() ? 1 : b[i - (nd - b.size())];
        if (da != db && da != 1 && db != 1)
            throw runtime_error("cannot broadcast shapes " + shape_str(a) + " and " + shape_str(b));
        out[i] = da == 1 ? db : da;
    }
    return out;
}

// Strides that read a contiguous tensor of shape `in` at the positions of
// `out`: its own row-major strides, and 0 wherever it is being broadcast.
// The same numbers are its row-major index for each output element, which
// is what the backward pass needs to sum a gradient back down.
static Shape broadcast_strides(const Shape& in, const Shape& out) {
    const size_t nd = out.size(), off = nd - in.size();
    const Shape rm = Tensor::contiguous_strides(in);
    Shape st(nd, 0);
    for (size_t i = off; i < nd; i++)
        st[i] = in[i - off] == 1 ? 0 : rm[i - off];
    return st;
}

// Calls f(i, ia, ib) for every output element: i its row-major index, ia and
// ib the matching elements of the two inputs. Rows (everything but the last
// dimension) are independent and run in parallel when `parallel` is set; a
// caller that scatters into a broadcast input must not set it.
template <class F>
static void broadcast_walk(const Shape& out, const Shape& sa, const Shape& sb,
                          bool parallel, F f) {
    const int nd = (int)out.size();
    if (nd == 0) { f(0, 0, 0); return; }
    const int64_t inner = out[nd - 1];
    int64_t total = 1;
    for (int64_t d : out) total *= d;
    if (total == 0) return;
    const int rows = (int)(total / inner);
    const int64_t sai = sa[nd - 1], sbi = sb[nd - 1];
    #pragma omp parallel for schedule(static) if(parallel && total >= 32768)
    for (int r = 0; r < rows; r++) {
        int64_t ia = 0, ib = 0, rr = r;
        for (int d = nd - 2; d >= 0; d--) {
            const int64_t i = rr % out[d];
            rr /= out[d];
            ia += i * sa[d];
            ib += i * sb[d];
        }
        const int64_t base = (int64_t)r * inner;
        for (int64_t j = 0; j < inner; j++)
            f(base + j, ia + j * sai, ib + j * sbi);
    }
}

// defined with the other higher-order helpers below
static TensorPtr sum_to(TensorPtr g, const Shape& shape);

enum class BinOp { Add, Sub, Mul, Div };

template <BinOp OP> static inline float bin_fwd(float x, float y) {
    if constexpr (OP == BinOp::Add) return x + y;
    else if constexpr (OP == BinOp::Sub) return x - y;
    else if constexpr (OP == BinOp::Mul) return x * y;
    else return x / y;
}
// dL/dx and dL/dy for one element, given the upstream gradient g
template <BinOp OP> static inline float bin_dx(float g, float, float y) {
    if constexpr (OP == BinOp::Add || OP == BinOp::Sub) return g;
    else if constexpr (OP == BinOp::Mul) return g * y;
    else return g / y;
}
template <BinOp OP> static inline float bin_dy(float g, float x, float y) {
    if constexpr (OP == BinOp::Add) return g;
    else if constexpr (OP == BinOp::Sub) return -g;
    else if constexpr (OP == BinOp::Mul) return g * x;
    else return -g * x / (y * y);
}

template <BinOp OP>
static TensorPtr binary_op(const char* name, TensorPtr a, TensorPtr b) {
    a = checked_input(name, a);
    b = checked_input(name, b);
    const Shape out_shape = broadcast_shapes(a->shape(), b->shape());
    const bool same = a->shape() == b->shape();

    if constexpr (OP == BinOp::Div) {
        const float* Bv = b->data();
        for (int64_t i = 0; i < b->numel(); i++)
            if (Bv[i] == 0.0f)
                throw runtime_error("Division by zero at index " + to_string(i));
    }

    auto result = make_shared<Tensor>(out_shape);
    const float* A = a->data();
    const float* Bv = b->data();
    float* C = result->data();

    if (same) {
        const int n = (int)result->numel();
        #pragma omp parallel for schedule(static) if(n >= 32768)
        for (int i = 0; i < n; i++) C[i] = bin_fwd<OP>(A[i], Bv[i]);
    } else {
        const Shape sa = broadcast_strides(a->shape(), out_shape);
        const Shape sb = broadcast_strides(b->shape(), out_shape);
        broadcast_walk(out_shape, sa, sb, true,
                       [&](int64_t i, int64_t ia, int64_t ib) { C[i] = bin_fwd<OP>(A[ia], Bv[ib]); });
    }

    if (a->requires_grad() || b->requires_grad()) {
        auto node = make_node(result, {a, b});
        node->name = name;
        node->backward_graph = [a, b](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> {
            TensorPtr ga, gb;
            if constexpr (OP == BinOp::Add)      { ga = G; gb = G; }
            else if constexpr (OP == BinOp::Sub) { ga = G; gb = multiply(-1.0f, G); }
            else if constexpr (OP == BinOp::Mul) { ga = multiply(G, b); gb = multiply(G, a); }
            else { ga = divide(G, b); gb = multiply(-1.0f, divide(multiply(G, a), multiply(b, b))); }
            return { a->requires_grad() ? sum_to(ga, a->shape()) : nullptr,
                     b->requires_grad() ? sum_to(gb, b->shape()) : nullptr };
        };
        node->backward_fn = [a, b, same, out_shape](const TensorPtr& result) {
            const float* G  = result->grad();
            const float* A  = a->data();
            const float* Bv = b->data();
            if (same) {
                const int n = (int)result->numel();
                if (a->requires_grad()) {
                    float* GA = a->grad();
                    #pragma omp parallel for schedule(static) if(n >= 32768)
                    for (int i = 0; i < n; i++) GA[i] += bin_dx<OP>(G[i], A[i], Bv[i]);
                }
                if (b->requires_grad()) {
                    float* GB = b->grad();
                    #pragma omp parallel for schedule(static) if(n >= 32768)
                    for (int i = 0; i < n; i++) GB[i] += bin_dy<OP>(G[i], A[i], Bv[i]);
                }
                return;
            }
            // A broadcast input received the same element many times, so its
            // gradient is the sum over every place it was used. Walking the
            // output and adding into its index does exactly that; it runs in
            // parallel only for an input that was not broadcast, where every
            // output element maps to a different slot.
            const Shape sa = broadcast_strides(a->shape(), out_shape);
            const Shape sb = broadcast_strides(b->shape(), out_shape);
            if (a->requires_grad()) {
                float* GA = a->grad();
                broadcast_walk(out_shape, sa, sb, a->shape() == out_shape,
                               [&](int64_t i, int64_t ia, int64_t ib) {
                                   GA[ia] += bin_dx<OP>(G[i], A[ia], Bv[ib]); });
            }
            if (b->requires_grad()) {
                float* GB = b->grad();
                broadcast_walk(out_shape, sa, sb, b->shape() == out_shape,
                               [&](int64_t i, int64_t ia, int64_t ib) {
                                   GB[ib] += bin_dy<OP>(G[i], A[ia], Bv[ib]); });
            }
        };
    }
    return result;
}

vector<int64_t> index_values(TensorPtr t, const char* op) {
    auto c = contiguous(t);
    vector<int64_t> out((size_t)c->numel());
    const void* base = c->storage()->data();
    for (int64_t i = 0; i < c->numel(); i++) {
        const double v = strided::load(base, c->dtype(), c->offset() + i);
        if (v != std::floor(v))
            throw runtime_error(string(op) + ": index " + to_string(v) + " is not an integer");
        out[(size_t)i] = (int64_t)v;
    }
    return out;
}

// ── Higher-order gradients: backward passes written with ops ────────────────
//
// Each op that supports it attaches a backward_graph: its gradient formula
// written with the differentiable ops in this file rather than raw loops. Run
// with gradients enabled, those ops record a graph of their own, so a
// gradient can itself be differentiated -- what MAML and Hessian-vector
// products need. autograd::grad drives them.

// Sum a gradient back down to `shape`, undoing a broadcast: over leading
// dims the target lacks, and over dims where it had size 1.
static TensorPtr sum_to(TensorPtr g, const Shape& shape) {
    if (g->shape() == shape) return g;
    TensorPtr r = g;
    const int lead = r->dim() - (int)shape.size();
    for (int i = 0; i < lead; i++) r = sum(r, 0);
    for (int i = 0; i < (int)shape.size() && i < r->dim(); i++)
        if (shape[i] == 1 && r->shape()[i] != 1) r = sum(r, i, true);
    if (r->shape() != shape) r = reshape(r, shape);
    return r;
}

// Broadcast a gradient up to `shape`: the backward of a reduction.
static TensorPtr expand_to(TensorPtr g, const Shape& shape) {
    if (g->shape() == shape) return g;
    return add(Tensor::zeros(shape), g);
}



TensorPtr add(TensorPtr a, TensorPtr b) { return binary_op<BinOp::Add>("add", a, b); }

// Scalar + tensor
TensorPtr add(float scalar, TensorPtr a) {
    a = checked_input("add", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_3 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_3 >= 32768)
    for (int i = 0; i < n_3; i++)
        result->data()[i] = scalar + a->data()[i];

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "add";
        node->backward_graph = [](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> { return {G}; };
        node->backward_fn = [a](const TensorPtr& result) {
            const int n_4 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_4 >= 32768)
            for (int i = 0; i < n_4; i++)
                a->grad()[i] += result->grad()[i];
        };
    }

    return result;
}

TensorPtr multiply(TensorPtr a, TensorPtr b) { return binary_op<BinOp::Mul>("multiply", a, b); }

// Scalar * tensor
TensorPtr multiply(float scalar, TensorPtr a) {
    a = checked_input("multiply", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_7 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_7 >= 32768)
    for (int i = 0; i < n_7; i++)
        result->data()[i] = scalar * a->data()[i];

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "multiply";
        node->backward_graph = [scalar](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> { return {multiply(scalar, G)}; };
        node->backward_fn = [a, scalar](const TensorPtr& result) {
            const int n_8 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_8 >= 32768)
            for (int i = 0; i < n_8; i++)
                a->grad()[i] += scalar * result->grad()[i];
        };
    }

    return result;
}

// a @ b for a [..., m, k] and b [..., k, n].
//
// The leading (batch) dimensions broadcast like any other op. When b is a
// plain matrix -- a weight -- every leading dimension of a folds into the row
// count and the whole thing is one gemm, which is the case that matters for
// speed. A b that is the transpose view of a contiguous matrix is multiplied
// with the transpose as a BLAS flag, so x @ W.transpose() copies nothing.
//
// This used to swap the operands when their shapes did not line up, which
// silently computed b @ a; mismatched shapes now raise.
TensorPtr matmul(TensorPtr a, TensorPtr b) {
    if (a->dim() < 2 || b->dim() < 2)
        throw runtime_error("matmul: operands must be at least 2-D, got " + shape_str(a->shape()) +
                            " and " + shape_str(b->shape()) + "; reshape a vector to [1, k] or [k, 1]");
    const int64_t m = a->size(-2), k = a->size(-1), n = b->size(-1);
    if (b->size(-2) != k)
        throw runtime_error("matmul: inner dimensions differ: " + shape_str(a->shape()) + " @ " +
                            shape_str(b->shape()));

    a = checked_input("matmul", a);

    if (b->dim() == 2) {
        // b viewed as W^T for a contiguous W [n, k]: strides [1, k]
        const bool b_is_wt = !b->is_contiguous() && b->dtype() == DType::Float32 &&
                             b->strides()[0] == 1 && b->strides()[1] == k;
        if (!b_is_wt) b = checked_input("matmul", b);

        const int rows = (int)(a->numel() / k);
        Shape out_shape = a->shape();
        out_shape.back() = n;
        auto result = make_shared<Tensor>(out_shape);
        if (b_is_wt) matmul_forward_bt(rows, (int)n, (int)k, a->data(), b->data(), result->data());
        else         matmul_forward(rows, (int)n, (int)k, a->data(), b->data(), result->data());

        if (a->requires_grad() || b->requires_grad()) {
            auto node = make_node(result, {a, b});
            node->name = "matmul";
            node->backward_graph = [a, b, rows, k, n](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> {
                TensorPtr ga, gb;
                if (a->requires_grad()) ga = matmul(G, transpose(b));
                if (b->requires_grad())
                    gb = matmul(transpose(reshape(a, {(int64_t)rows, k})), reshape(G, {(int64_t)rows, n}));
                return {ga, gb};
            };
            node->backward_fn = [a, b, rows, n, k, b_is_wt](const TensorPtr& result) {
                const float* g = result->grad();
                if (a->requires_grad()) {
                    if (b_is_wt) matmul_accumulate(rows, (int)k, (int)n, g, b->data(), a->grad());
                    else matmul_backward_a(rows, (int)k, (int)n, g, b->data(), a->grad());
                }
                // b's gradient is row-major over its own [k, n] shape either way
                if (b->requires_grad())
                    matmul_backward_b(rows, (int)k, (int)n, g, a->data(), b->grad());
            };
        }
        return result;
    }

    // batched: one gemm per pair of matrices, batch dims broadcast
    b = checked_input("matmul", b);
    const Shape batch_a(a->shape().begin(), a->shape().end() - 2);
    const Shape batch_b(b->shape().begin(), b->shape().end() - 2);
    const Shape batch = broadcast_shapes(batch_a, batch_b);
    Shape out_shape = batch;
    out_shape.push_back(m);
    out_shape.push_back(n);
    auto result = make_shared<Tensor>(out_shape);

    // index of each output matrix's a and b matrices
    int64_t nb = 1;
    for (int64_t d : batch) nb *= d;
    vector<int64_t> ia((size_t)nb), ib((size_t)nb);
    {
        const Shape sa = broadcast_strides(batch_a, batch), sb = broadcast_strides(batch_b, batch);
        if (batch.empty()) { ia[0] = ib[0] = 0; }
        else broadcast_walk(batch, sa, sb, false,
                            [&](int64_t i, int64_t x, int64_t y) { ia[i] = x; ib[i] = y; });
    }
    const int64_t sz_a = m * k, sz_b = k * n, sz_c = m * n;
    for (int64_t i = 0; i < nb; i++)
        matmul_forward((int)m, (int)n, (int)k, a->data() + ia[i] * sz_a, b->data() + ib[i] * sz_b,
                       result->data() + i * sz_c);

    if (a->requires_grad() || b->requires_grad()) {
        auto node = make_node(result, {a, b});
        node->name = "matmul";
        node->backward_graph = [a, b](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> {
            TensorPtr ga, gb;
            if (a->requires_grad()) ga = sum_to(matmul(G, transpose(b)), a->shape());
            if (b->requires_grad()) gb = sum_to(matmul(transpose(a), G), b->shape());
            return {ga, gb};
        };
        node->backward_fn = [a, b, ia, ib, nb, m, n, k, sz_a, sz_b, sz_c](const TensorPtr& result) {
            const float* g = result->grad();
            // serial over the batch: a broadcast operand accumulates from
            // several output matrices into the same gradient
            for (int64_t i = 0; i < nb; i++) {
                if (a->requires_grad())
                    matmul_backward_a((int)m, (int)k, (int)n, g + i * sz_c, b->data() + ib[i] * sz_b,
                                      a->grad() + ia[i] * sz_a);
                if (b->requires_grad())
                    matmul_backward_b((int)m, (int)k, (int)n, g + i * sz_c, a->data() + ia[i] * sz_a,
                                      b->grad() + ib[i] * sz_b);
            }
        };
    }
    return result;
}

// ReLU activation: max(0, x) for each element
TensorPtr relu(TensorPtr a) {
    a = checked_input("relu", a);

    auto result = make_shared<Tensor>(a->shape());
    const int n_9 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_9 >= 32768)
    for (int i = 0; i < n_9; i++){
        result->data()[i] = a->data()[i] > 0 ? a->data()[i] : 0.0f;
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "relu";
        node->backward_graph = [a](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> {
            // the mask is a constant: relu is piecewise linear, so its second
            // derivative is zero wherever it exists
            auto mask = make_shared<Tensor>(a->shape());
            const float* A = a->data();
            float* M = mask->data();
            for (int64_t i = 0; i < a->numel(); i++) M[i] = A[i] > 0.0f ? 1.0f : 0.0f;
            return {multiply(G, mask)};
        };
        node->backward_fn = [a](const TensorPtr& result) {
            const int n_10 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_10 >= 32768)
            for (int i = 0; i < n_10; i++)
                if (a->data()[i] > 0) a->grad()[i] += result->grad()[i];
        };
    }

    return result;
}

// Sigmoid activation: 1 / (1 + e^-x) for each element
TensorPtr sigmoid(TensorPtr a) {
    a = checked_input("sigmoid", a);

    auto result = make_shared<Tensor>(a->shape());
    const int n_11 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_11 >= 32768)
    for (int i = 0; i < n_11; i++){
        result->data()[i] = 1.0f / (1.0f + exp(-a->data()[i]));
    }

    if (a->requires_grad()){
        auto node = make_node(result, {a});
        node->name = "sigmoid";
        node->backward_graph = [](const TensorPtr& out, const TensorPtr& G) -> std::vector<TensorPtr> { return {multiply(G, multiply(out, subtract(1.0f, out)))}; };

        node->backward_fn = [a](const TensorPtr& result){
            const int n_12 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_12 >= 32768)
            for (int i = 0; i < n_12; i++){
                a->grad()[i] += result->data()[i] * (1 - result->data()[i]) * result->grad()[i];
            }
        };
    }

    return result;
}

TensorPtr subtract(TensorPtr a, TensorPtr b) { return binary_op<BinOp::Sub>("subtract", a, b); }

// Scalar - tensor  (d/da = -1)
TensorPtr subtract(float scalar, TensorPtr a) {
    a = checked_input("subtract", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_15 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_15 >= 32768)
    for (int i = 0; i < n_15; i++)
        result->data()[i] = scalar - a->data()[i];

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "subtract";
        node->backward_graph = [](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> { return {multiply(-1.0f, G)}; };
        node->backward_fn = [a](const TensorPtr& result) {
            const int n_16 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_16 >= 32768)
            for (int i = 0; i < n_16; i++)
                a->grad()[i] -= result->grad()[i];
        };
    }

    return result;
}

// Tensor - scalar  (d/da = 1)
TensorPtr subtract(TensorPtr a, float scalar) {
    a = checked_input("subtract", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_17 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_17 >= 32768)
    for (int i = 0; i < n_17; i++)
        result->data()[i] = a->data()[i] - scalar;

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "subtract";
        node->backward_graph = [](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> { return {G}; };
        node->backward_fn = [a](const TensorPtr& result) {
            const int n_18 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_18 >= 32768)
            for (int i = 0; i < n_18; i++)
                a->grad()[i] += result->grad()[i];
        };
    }

    return result;
}

TensorPtr divide(TensorPtr a, TensorPtr b) { return binary_op<BinOp::Div>("divide", a, b); }

// Scalar / tensor  dL/da = -scalar * result.grad / a^2
TensorPtr divide(float scalar, TensorPtr a) {
    a = checked_input("divide", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_21 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_21 >= 32768)
    for (int i = 0; i < n_21; i++) {
        if (a->data()[i] == 0)
            throw std::runtime_error("Division by zero at index " + std::to_string(i));
        result->data()[i] = scalar / a->data()[i];
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "divide";
        node->backward_graph = [a, scalar](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> { return {multiply(-scalar, divide(G, multiply(a, a)))}; };
        node->backward_fn = [a, scalar](const TensorPtr& result) {
            const int n_22 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_22 >= 32768)
            for (int i = 0; i < n_22; i++)
                a->grad()[i] += -scalar * result->grad()[i] / (a->data()[i] * a->data()[i]);
        };
    }

    return result;
}

// Tensor / scalar  dL/da = result.grad / scalar
TensorPtr divide(TensorPtr a, float scalar) {
    a = checked_input("divide", a);
    if (scalar == 0)
        throw std::runtime_error("Division by zero");
    auto result = make_shared<Tensor>(a->shape());
    const int n_23 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_23 >= 32768)
    for (int i = 0; i < n_23; i++)
        result->data()[i] = a->data()[i] / scalar;

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "divide";
        node->backward_graph = [scalar](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> { return {multiply(1.0f / scalar, G)}; };
        node->backward_fn = [a, scalar](const TensorPtr& result) {
            const int n_24 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_24 >= 32768)
            for (int i = 0; i < n_24; i++)
                a->grad()[i] += result->grad()[i] / scalar;
        };
    }

    return result;
}

// Tanh activation: (e^x - e^-x) / (e^x + e^-x)
TensorPtr tanh_op(TensorPtr a) {
    a = checked_input("tanh_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_25 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_25 >= 32768)
    for (int i = 0; i < n_25; i++){
        result->data()[i] = (exp(a->data()[i]) - exp(-a->data()[i])) / (exp(a->data()[i]) + exp(-a->data()[i]));
    }

    if (a->requires_grad()){
        auto node = make_node(result, {a});
        node->name = "tanh_op";
        node->backward_graph = [](const TensorPtr& out, const TensorPtr& G) -> std::vector<TensorPtr> { return {multiply(G, subtract(1.0f, multiply(out, out)))}; };

        node->backward_fn = [a](const TensorPtr& result){
            const int n_26 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_26 >= 32768)
            for (int i = 0; i < n_26; i++){
                a->grad()[i] += (1 - result->data()[i] * result->data()[i]) * result->grad()[i];
            }
        };
    }

    return result;
}

// Softmax over the last dimension, for any rank.
//
// Subtracting the row maximum before exponentiating changes nothing
// mathematically -- softmax(x + c) = softmax(x) -- and keeps every exponent at
// or below zero. Without it a logit of 90 already overflows to inf and the
// result is NaN, which attention scores and an untrained output layer reach.
TensorPtr softmax(TensorPtr a) {
    a = checked_input("softmax", a);
    if (a->dim() < 1) throw runtime_error("softmax: needs at least one dimension");
    const int64_t cols = a->shape().back();
    const int rows = (int)(cols ? a->numel() / cols : 0);
    auto result = make_shared<Tensor>(a->shape());
    const float* A = a->data();
    float* R = result->data();

    #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 32768)
    for (int i = 0; i < rows; i++) {
        const float* x = A + (int64_t)i * cols;
        float* y = R + (int64_t)i * cols;
        float mx = x[0];
        for (int64_t j = 1; j < cols; j++) mx = max(mx, x[j]);
        double total = 0.0;
        for (int64_t j = 0; j < cols; j++) { y[j] = std::exp(x[j] - mx); total += y[j]; }
        const float inv = (float)(1.0 / total);
        for (int64_t j = 0; j < cols; j++) y[j] *= inv;
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "softmax";
        node->backward_graph = [](const TensorPtr& out, const TensorPtr& G) -> std::vector<TensorPtr> {
            // dx = y * (g - sum(g * y)) over the last dim
            auto s = sum(multiply(G, out), out->dim() - 1, true);
            return {multiply(out, subtract(G, s))};
        };
        node->backward_fn = [a, rows, cols](const TensorPtr& result) {
            // dL/dx_j = y_j * (g_j - sum_k g_k y_k), per row
            const float* Y = result->data();
            const float* G = result->grad();
            float* GA = a->grad();
            #pragma omp parallel for schedule(static) if((int64_t)rows * cols >= 32768)
            for (int i = 0; i < rows; i++) {
                const float* y = Y + (int64_t)i * cols;
                const float* g = G + (int64_t)i * cols;
                float dot = 0.0f;
                for (int64_t j = 0; j < cols; j++) dot += g[j] * y[j];
                float* ga = GA + (int64_t)i * cols;
                for (int64_t j = 0; j < cols; j++) ga[j] += y[j] * (g[j] - dot);
            }
        };
    }
    return result;
}

// Element-wise natural log
TensorPtr log_op(TensorPtr a) {
    a = checked_input("log_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_27 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_27 >= 32768)
    for (int i = 0; i < n_27; i++){
        if (a->data()[i] <= 0)
            throw runtime_error("Log of non-positive value at index " + to_string(i));
        result->data()[i] = log(a->data()[i]);
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "log_op";
        node->backward_graph = [a](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> { return {divide(G, a)}; };
        node->backward_fn = [a](const TensorPtr& result) {
            const int n_28 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_28 >= 32768)
            for (int i = 0; i < n_28; i++)
                a->grad()[i] += result->grad()[i] / a->data()[i];  // d(ln x)/dx = 1/x
        };
    }

    return result;
}

// Element-wise e^x
TensorPtr exp_op(TensorPtr a) {
    a = checked_input("exp_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_29 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_29 >= 32768)
    for (int i = 0; i < n_29; i++){
        result->data()[i] = exp(a->data()[i]);
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "exp_op";
        node->backward_graph = [](const TensorPtr& out, const TensorPtr& G) -> std::vector<TensorPtr> { return {multiply(G, out)}; };
        node->backward_fn = [a](const TensorPtr& result) {
            const int n_30 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_30 >= 32768)
            for (int i = 0; i < n_30; i++)
                a->grad()[i] += result->data()[i] * result->grad()[i];  // d(e^x)/dx = e^x = result
        };
    }

    return result;
}

// Element-wise power: x^p
TensorPtr pow_op(TensorPtr a, float p) {
    a = checked_input("pow_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_31 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_31 >= 32768)
    for (int i = 0; i < n_31; i++){
        result->data()[i] = pow(a->data()[i], p);
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "pow_op";
        node->backward_graph = [a, p](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> { return {multiply(G, multiply(p, pow_op(a, p - 1.0f)))}; };
        node->backward_fn = [a, p](const TensorPtr& result) {
            const int n_32 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_32 >= 32768)
            for (int i = 0; i < n_32; i++)
                a->grad()[i] += p * pow(a->data()[i], p - 1) * result->grad()[i];  // d(x^p)/dx = p*x^(p-1)
        };
    }

    return result;
}

// Element-wise square root
TensorPtr sqrt_op(TensorPtr a) {
    a = checked_input("sqrt_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_33 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_33 >= 32768)
    for (int i = 0; i < n_33; i++){
        if (a->data()[i] < 0)
            throw runtime_error("Cannot compute square root of negative value " +
                                to_string(a->data()[i]) + " at index " + to_string(i));
        result->data()[i] = sqrt(a->data()[i]);
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "sqrt_op";
        node->backward_graph = [](const TensorPtr& out, const TensorPtr& G) -> std::vector<TensorPtr> { return {divide(G, multiply(2.0f, out))}; };
        node->backward_fn = [a](const TensorPtr& result) {
            const int n_34 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_34 >= 32768)
            for (int i = 0; i < n_34; i++)
                a->grad()[i] += result->grad()[i] / (2.0f * result->data()[i]);  // d(sqrt(x))/dx = 1/(2*sqrt(x))
        };
    }

    return result;
}

// Element-wise absolute value
TensorPtr abs_op(TensorPtr a) {
    a = checked_input("abs_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_35 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_35 >= 32768)
    for (int i = 0; i < n_35; i++){
        result->data()[i] = abs(a->data()[i]);
    }

    if (a->requires_grad()){
        auto node = make_node(result, {a});
        node->name = "abs_op";

        node->backward_fn = [a](const TensorPtr& result){
            const int n_36 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_36 >= 32768)
            for (int i = 0; i < n_36; i++){
                if (a->data()[i] > 0) a->grad()[i] += result->grad()[i];
                else if (a->data()[i] < 0) a->grad()[i] -= result->grad()[i];
            }
        };
    }

    return result;
}

// Reduction over one dimension, or over everything.
//
// dim = -1 reduces every element to a [1] tensor -- unlike PyTorch, where -1
// means the last dimension. Existing code, the gradcheck harness included,
// relies on -1 meaning "all", and quietly changing it would change numbers
// without raising. Any other negative dim is refused rather than guessed at.
static TensorPtr reduce_sum(const char* name, TensorPtr a, int dim, bool keepdim, bool mean) {
    a = checked_input(name, a);
    const int nd = a->dim();

    if (dim == -1) {
        Shape out_shape = keepdim ? Shape(max(nd, 1), 1) : Shape{1};
        auto result = make_shared<Tensor>(out_shape);
        const int64_t n = a->numel();
        double acc = 0.0;                       // a float sum of millions drifts
        const float* A = a->data();
        for (int64_t i = 0; i < n; i++) acc += A[i];
        result->data()[0] = (float)(mean ? acc / (double)n : acc);
        if (a->requires_grad()) {
            auto node = make_node(result, {a});
            node->name = name;
            node->backward_graph = [a, n, mean](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> {
                auto g = expand_to(reshape(G, {1}), a->shape());
                return {mean ? multiply(1.0f / (float)n, g) : g};
            };
            node->backward_fn = [a, n, mean](const TensorPtr& result) {
                const float g = result->grad()[0] / (mean ? (float)n : 1.0f);
                float* GA = a->grad();
                for (int64_t i = 0; i < n; i++) GA[i] += g;
            };
        }
        return result;
    }

    if (dim < 0 || dim >= nd)
        throw runtime_error(string(name) + ": dim " + to_string(dim) + " is out of range for a " +
                            to_string(nd) + "-D tensor (-1 reduces everything)");

    // view as [outer, len, inner] and reduce the middle
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < dim; d++) outer *= a->shape()[d];
    for (int d = dim + 1; d < nd; d++) inner *= a->shape()[d];
    const int64_t len = a->shape()[dim];

    Shape out_shape;
    for (int d = 0; d < nd; d++)
        if (d != dim) out_shape.push_back(a->shape()[d]);
        else if (keepdim) out_shape.push_back(1);
    if (out_shape.empty()) out_shape = {1};

    auto result = make_shared<Tensor>(out_shape);
    const float* A = a->data();
    float* R = result->data();
    const float scale = mean ? 1.0f / (float)len : 1.0f;
    for (int64_t o = 0; o < outer; o++)
        for (int64_t k = 0; k < len; k++) {
            const float* row = A + (o * len + k) * inner;
            float* out = R + o * inner;
            for (int64_t j = 0; j < inner; j++) out[j] += row[j];
        }
    if (mean) for (int64_t i = 0; i < result->numel(); i++) R[i] *= scale;

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = name;
        node->backward_graph = [a, dim, scale](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> {
            Shape kept = a->shape();
            kept[dim] = 1;
            auto g = expand_to(reshape(G, kept), a->shape());
            return {scale != 1.0f ? multiply(scale, g) : g};
        };
        node->backward_fn = [a, outer, len, inner, scale](const TensorPtr& result) {
            const float* G = result->grad();
            float* GA = a->grad();
            for (int64_t o = 0; o < outer; o++)
                for (int64_t k = 0; k < len; k++)
                    for (int64_t j = 0; j < inner; j++)
                        GA[(o * len + k) * inner + j] += G[o * inner + j] * scale;
        };
    }
    return result;
}

TensorPtr sum(TensorPtr a, int dim, bool keepdim)  { return reduce_sum("sum", a, dim, keepdim, false); }

TensorPtr mean(TensorPtr a, int dim, bool keepdim) { return reduce_sum("mean", a, dim, keepdim, true); }

// Max element — returns a scalar tensor
// Max element. The gradient reaches only the element that attained the
// maximum; every other element has no influence on the result. Ties go to the
// first occurrence, matching the forward scan.
TensorPtr max_op(TensorPtr a) {
    a = checked_input("max_op", a);
    auto result = make_shared<Tensor>(Shape{1});
    result->data()[0] = a->data()[0];
    int arg = 0;
    for (int i = 1; i < a->num_el(); i++)
        if (a->data()[i] > result->data()[0]) {
            result->data()[0] = a->data()[i];
            arg = i;
        }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "max_op";
        node->backward_fn = [a, arg](const TensorPtr& result) {
            a->grad()[arg] += result->grad()[0];
        };
    }

    return result;
}

// Min element — returns a scalar tensor
// Min element — mirror of max_op, gradient reaches only the arg-min.
TensorPtr min_op(TensorPtr a) {
    a = checked_input("min_op", a);
    auto result = make_shared<Tensor>(Shape{1});
    result->data()[0] = a->data()[0];
    int arg = 0;
    for (int i = 1; i < a->num_el(); i++)
        if (a->data()[i] < result->data()[0]) {
            result->data()[0] = a->data()[i];
            arg = i;
        }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "min_op";
        node->backward_fn = [a, arg](const TensorPtr& result) {
            a->grad()[arg] += result->grad()[0];
        };
    }

    return result;
}

// Clip — clamp all values between min and max
TensorPtr clip(TensorPtr a, float min_val, float max_val) {
    a = checked_input("clip", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_41 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_41 >= 32768)
    for (int i = 0; i < n_41; i++){
        if (a->data()[i] > max_val){
            result->data()[i] = max_val;
        } else if (a->data()[i] < min_val) {
            result->data()[i] = min_val;
        } else {
            result->data()[i] = a->data()[i];
        }
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "clip";
        node->backward_fn = [a, min_val, max_val](const TensorPtr& result) {
            const int n_42 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_42 >= 32768)
            for (int i = 0; i < n_42; i++) {
                // gradient is 1 where value wasn't clamped, 0 where it was
                if (a->data()[i] > min_val && a->data()[i] < max_val)
                    a->grad()[i] += result->grad()[i];
            }
        };
    }

    return result;
}


// Index of the largest value along the last dimension, for any rank. The
// result drops that dimension ([rows, cols] -> [rows]) and holds the indices
// as float32, which is what every caller here reads.
TensorPtr argmax(TensorPtr a) {
    a = checked_input("argmax", a);
    if (a->dim() < 1) throw runtime_error("argmax: needs at least one dimension");
    const int64_t cols = a->shape().back();
    const int64_t rows = cols ? a->numel() / cols : 0;
    Shape out_shape(a->shape().begin(), a->shape().end() - 1);
    if (out_shape.empty()) out_shape = {1};
    auto result = make_shared<Tensor>(out_shape);
    const float* A = a->data();
    for (int64_t i = 0; i < rows; i++) {
        const float* x = A + i * cols;
        int64_t best = 0;
        for (int64_t j = 1; j < cols; j++)
            if (x[j] > x[best]) best = j;
        result->data()[i] = (float)best;
    }
    return result;
}

// GELU activation — tanh approximation used by GPT-2/3/4.
// Formula: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
TensorPtr gelu(TensorPtr a) {
    a = checked_input("gelu", a);
    auto result = make_shared<Tensor>(a->shape());
    int n = a->num_el();
    constexpr float kSqrt2OverPi = 0.7978845608f;

    // gelu was missed when the other elementwise ops were parallelised -- it
    // bounds its loop with a local, not num_el() -- and tanhf is an opaque
    // call that kept it to one lane besides
    #pragma omp parallel for schedule(static) if(n >= 32768)
    for (int i = 0; i < n; i++) {
        float x = a->data()[i];
        float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
        result->data()[i] = 0.5f * x * (1.0f + fastmath::tanh(inner));
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->name = "gelu";
        node->backward_graph = [a](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> {
            // d/dx of 0.5 x (1 + tanh(k (x + c x^3))), built from ops
            const float k = 0.7978845608f, c = 0.044715f;
            auto a2    = multiply(a, a);
            auto t     = tanh_op(multiply(k, add(a, multiply(c, multiply(a2, a)))));
            auto left  = multiply(0.5f, add(1.0f, t));
            auto sech2 = subtract(1.0f, multiply(t, t));
            auto inner = multiply(k, add(1.0f, multiply(3.0f * c, a2)));
            auto right = multiply(0.5f, multiply(a, multiply(sech2, inner)));
            return {multiply(G, add(left, right))};
        };
        node->backward_fn = [a](const TensorPtr& result) {
            constexpr float k = 0.7978845608f;
            int m = a->num_el();
            #pragma omp parallel for schedule(static) if(m >= 32768)
            for (int i = 0; i < m; i++) {
                float x = a->data()[i];
                float inner = k * (x + 0.044715f * x * x * x);
                float t = fastmath::tanh(inner);
                float d_inner = k * (1.0f + 3.0f * 0.044715f * x * x);
                float grad = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * d_inner;
                a->grad()[i] += result->grad()[i] * grad;
            }
        };
    }

    return result;
}

TensorPtr concat_op(vector<TensorPtr> inputs, int axis) {
    for (auto& t_ : inputs) t_ = checked_input("concat_op", t_);
    if (inputs.empty())
        throw runtime_error("concat_op: no inputs");

    const Shape shape0 = inputs[0]->shape();
    const int ndim = (int)shape0.size();
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim)
        throw runtime_error("concat_op: axis out of range");

    // every input must match on all axes except the concatenation one
    int concat_total = 0;
    for (auto& t : inputs) {
        if ((int)t->shape().size() != ndim)
            throw runtime_error("concat_op: inputs differ in rank");
        for (int d = 0; d < ndim; d++)
            if (d != axis && t->shape()[d] != shape0[d])
                throw runtime_error("concat_op: inputs differ on axis " + to_string(d));
        concat_total += t->shape()[axis];
    }

    Shape out_shape = shape0;
    out_shape[axis] = concat_total;
    auto result = make_shared<Tensor>(out_shape);

    // Treat the tensor as [outer, axis_dim, inner]: copying a contiguous
    // inner-sized run per outer index per input is all that is needed.
    int outer = 1;
    for (int d = 0; d < axis; d++) outer *= shape0[d];
    int inner = 1;
    for (int d = axis + 1; d < ndim; d++) inner *= shape0[d];

    bool any_grad = false;
    for (auto& t : inputs) if (t->requires_grad()) any_grad = true;

    for (int o = 0; o < outer; o++) {
        int written = 0;
        for (auto& t : inputs) {
            const int span = t->shape()[axis] * inner;
            const float* src = t->data() + o * span;
            float* dst = result->data() + o * concat_total * inner + written;
            for (int i = 0; i < span; i++) dst[i] = src[i];
            written += span;
        }
    }

    if (any_grad) {
        auto node = make_node(result, inputs);
        node->name = "concat_op";
        node->backward_fn = [inputs, outer, inner, concat_total, axis](const TensorPtr& result) {
            for (int o = 0; o < outer; o++) {
                int read = 0;
                for (auto& t : inputs) {
                    const int span = t->shape()[axis] * inner;
                    if (t->requires_grad()) {
                        const float* src = result->grad() + o * concat_total * inner + read;
                        float* dst = t->grad() + o * span;
                        for (int i = 0; i < span; i++) dst[i] += src[i];
                    }
                    read += span;
                }
            }
        };
    }

    return result;
}

TensorPtr matmul_bt(TensorPtr a, TensorPtr b) {
    a = checked_input("matmul_bt", a);
    b = checked_input("matmul_bt", b);
    if (a->shape().size() != 2 || b->shape().size() != 2)
        throw runtime_error("matmul_bt: both operands must be 2-D");

    const int m = a->shape()[0], k = a->shape()[1];
    const int n = b->shape()[0];
    if (b->shape()[1] != k)
        throw runtime_error("matmul_bt: a.cols must equal b.cols, got "
                            + to_string(k) + " and " + to_string(b->shape()[1]));

    auto result = make_shared<Tensor>(Shape{m, n});
    matmul_forward_bt(m, n, k, a->data(), b->data(),
                      result->data());

    if (a->requires_grad() || b->requires_grad()) {
        auto node = make_node(result, {a, b});
        node->name = "matmul_bt";

        node->backward_fn = [a, b, m, n, k](const TensorPtr& result) {
            const float* g = result->grad();

            // C = A @ B^T  =>  dA = dC @ B,  dB = dC^T @ A
            if (a->requires_grad())
                matmul_accumulate(m, k, n, g, b->data(), a->grad());

            if (b->requires_grad())
                matmul_backward_b(m, n, k, a->data(), g, b->grad());
        };
    }

    return result;
}

// ── Views ────────────────────────────────────────────────────────────────────
//
// A view shares its source's storage and differs only in shape, strides and
// offset, so taking one is O(1) and writes through it reach the source.
// Gradients follow the convention in tensor.hpp -- every grad buffer is
// row-major over its own tensor's shape -- so a view's backward only has to
// map positions between two row-major layouts.

static int wrap_dim(int d, int nd, const char* op) {
    const int w = d < 0 ? d + nd : d;
    if (w < 0 || w >= nd)
        throw runtime_error(string(op) + ": dimension " + to_string(d) +
                            " is out of range for a " + to_string(nd) + "-D tensor");
    return w;
}

TensorPtr permute(TensorPtr a, vector<int> dims) {
    const int nd = a->dim();
    if ((int)dims.size() != nd)
        throw runtime_error("permute: " + to_string(dims.size()) + " dims given for a " +
                            to_string(nd) + "-D tensor");
    vector<bool> seen(nd, false);
    for (int& d : dims) {
        d = wrap_dim(d, nd, "permute");
        if (seen[d]) throw runtime_error("permute: dimension repeated");
        seen[d] = true;
    }

    Shape shape(nd), strides(nd);
    for (int i = 0; i < nd; i++) {
        shape[i]   = a->shape()[dims[i]];
        strides[i] = a->strides()[dims[i]];
    }
    auto v = make_shared<Tensor>(a->storage(), shape, strides, a->offset());

    if (a->requires_grad()) {
        auto node = make_node(v, {a});
        node->name = "permute";
        node->backward_graph = [dims](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> {
            vector<int> inv(dims.size());
            for (size_t i = 0; i < dims.size(); i++) inv[dims[i]] = (int)i;
            return {permute(G, inv)};
        };
        node->backward_fn = [a, dims](const TensorPtr& v) {
            // Walking the view in row-major order with these strides lands on
            // each element's row-major position in `a`.
            const Shape a_rm = Tensor::contiguous_strides(a->shape());
            Shape to_a(dims.size());
            for (size_t i = 0; i < dims.size(); i++) to_a[i] = a_rm[dims[i]];
            const float* gv = v->grad();
            float* ga = a->grad();
            strided::for_each(v->shape(), to_a, 0,
                              [&](int64_t i, int64_t j) { ga[j] += gv[i]; });
        };
    }
    return v;
}

TensorPtr transpose(TensorPtr a, int dim0, int dim1) {
    const int nd = a->dim();
    vector<int> dims(nd);
    for (int i = 0; i < nd; i++) dims[i] = i;
    swap(dims[wrap_dim(dim0, nd, "transpose")], dims[wrap_dim(dim1, nd, "transpose")]);
    return permute(a, dims);
}

TensorPtr reshape(TensorPtr a, ShapeArg new_shape_arg) {
    Shape shape = new_shape_arg.get();

    // at most one -1, inferred from the rest
    int infer = -1;
    int64_t known = 1;
    for (int i = 0; i < (int)shape.size(); i++) {
        if (shape[i] == -1) {
            if (infer >= 0) throw runtime_error("reshape: only one dimension can be -1");
            infer = i;
        } else {
            known *= shape[i];
        }
    }
    if (infer >= 0) {
        if (known == 0 || a->numel() % known)
            throw runtime_error("reshape: cannot infer the -1 dimension");
        shape[infer] = a->numel() / known;
        known *= shape[infer];
    }
    if (known != a->numel())
        throw runtime_error("reshape: cannot reshape " + to_string(a->numel()) +
                            " elements into a shape of " + to_string(known));

    // a non-contiguous source has no single stride pattern for the new shape,
    // so it is materialised first; the copy is part of the graph
    TensorPtr src = a->is_contiguous() ? a : contiguous(a);
    auto v = make_shared<Tensor>(src->storage(), shape, Tensor::contiguous_strides(shape),
                                 src->offset());

    if (src->requires_grad()) {
        auto node = make_node(v, {src});
        node->name = "reshape";
        node->backward_graph = [src](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> { return {reshape(G, src->shape())}; };
        node->backward_fn = [src](const TensorPtr& v) {
            // element order is unchanged, so the gradient maps one to one
            const int64_t n = src->numel();
            const float* gv = v->grad();
            float* gs = src->grad();
            for (int64_t i = 0; i < n; i++) gs[i] += gv[i];
        };
    }
    return v;
}

TensorPtr contiguous(TensorPtr a) {
    if (a->is_contiguous())
        return a;

    auto out = make_shared<Tensor>(a->shape(), a->dtype());
    const size_t es = dtype_size(a->dtype());
    const unsigned char* src = static_cast<const unsigned char*>(a->storage()->data());
    unsigned char* dst = static_cast<unsigned char*>(out->storage()->data());
    strided::for_each(a->shape(), a->strides(), a->offset(), [&](int64_t i, int64_t off) {
        memcpy(dst + i * es, src + off * es, es);
    });

    if (a->requires_grad()) {
        auto node = make_node(out, {a});
        node->name = "contiguous";
        node->backward_graph = [](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> { return {G}; };
        node->backward_fn = [a](const TensorPtr& out) {
            // both gradients are row-major over the same shape
            const int64_t n = a->numel();
            const float* go = out->grad();
            float* ga = a->grad();
            for (int64_t i = 0; i < n; i++) ga[i] += go[i];
        };
    }
    return out;
}

// ── Fused linear ──────────────────────────────────────────────────────────────

TensorPtr linear_op(TensorPtr x, TensorPtr W, TensorPtr bias, TensorPtr residual) {
    x = checked_input("linear_op", x);
    W = checked_input("linear_op", W);
    bias = checked_input("linear_op", bias);
    residual = checked_input("linear_op", residual);

    if (x->dim() < 1 || W->dim() != 2 || x->shape().back() != W->shape()[0])
        throw runtime_error("linear_op: x must be [..., in] and W [in, out], got " +
                            shape_str(x->shape()) + " and " + shape_str(W->shape()));
    const int in = (int)W->shape()[0], out = (int)W->shape()[1];
    const int rows = (int)(x->numel() / in);
    Shape out_shape = x->shape();
    out_shape.back() = out;
    if (bias && bias->num_el() != out)
        throw runtime_error("linear_op: bias must have " + to_string(out) + " elements");
    if (residual && residual->shape() != out_shape)
        throw runtime_error("linear_op: residual must be " + shape_str(out_shape) +
                            ", got " + shape_str(residual->shape()));

    auto result = make_shared<Tensor>(out_shape);
    float* y = result->data();
    const float* rd = residual ? residual->data() : nullptr;
    const float* bd = bias ? bias->data() : nullptr;

    // seed with residual + bias, one pass, then the gemm accumulates onto it
    #pragma omp parallel for schedule(static) if((long long)rows * out >= 32768)
    for (int r = 0; r < rows; r++) {
        float* yr = y + (size_t)r * out;
        if (rd) {
            const float* rr = rd + (size_t)r * out;
            for (int j = 0; j < out; j++) yr[j] = rr[j];
        }
        if (bd)
            for (int j = 0; j < out; j++) yr[j] += bd[j];
    }
    matmul_accumulate(rows, out, in, x->data(), W->data(), y);

    vector<TensorPtr> inputs = {x, W};
    if (bias)     inputs.push_back(bias);
    if (residual) inputs.push_back(residual);
    bool need = false;
    for (auto& t : inputs) need = need || t->requires_grad();

    if (need) {
        auto node = make_node(result, inputs);
        node->name = "linear";
        node->backward_graph = [x, W, bias, residual, rows, in, out](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> {
            std::vector<TensorPtr> g;
            g.push_back(x->requires_grad() ? matmul(G, transpose(W)) : nullptr);
            g.push_back(W->requires_grad()
                ? matmul(transpose(reshape(x, {(int64_t)rows, (int64_t)in})), reshape(G, {(int64_t)rows, (int64_t)out}))
                : nullptr);
            if (bias)     g.push_back(bias->requires_grad() ? sum_to(G, bias->shape()) : nullptr);
            if (residual) g.push_back(residual->requires_grad() ? G : nullptr);
            return g;
        };
        node->backward_fn = [x, W, bias, residual, rows, in, out](const TensorPtr& result) {
            const float* g = result->grad();

            if (x->requires_grad())
                matmul_backward_a(rows, in, out, g, W->data(), x->grad());
            if (W->requires_grad())
                matmul_backward_b(rows, in, out, g, x->data(), W->grad());

            // bias: sum over rows, parallel over the output feature so each
            // thread owns its own slot
            if (bias && bias->requires_grad()) {
                #pragma omp parallel for schedule(static) if(out >= 256)
                for (int j = 0; j < out; j++) {
                    float acc = 0.0f;
                    for (int r = 0; r < rows; r++) acc += g[(size_t)r * out + j];
                    bias->grad()[j] += acc;
                }
            }

            // residual: the gradient passes straight through
            if (residual && residual->requires_grad()) {
                const int n = rows * out;
                #pragma omp parallel for schedule(static) if(n >= 32768)
                for (int i = 0; i < n; i++) residual->grad()[i] += g[i];
            }
        };
    }
    return result;
}

// ── autograd::grad ───────────────────────────────────────────────────────────

namespace autograd {

std::vector<TensorPtr> grad(TensorPtr output, std::vector<TensorPtr> inputs,
                            bool create_graph, TensorPtr grad_output) {
    if (!output->grad_fn())
        throw runtime_error("grad(): the output is not part of a graph -- it is a leaf, or was "
                            "computed with gradients disabled");
    if (!grad_output) {
        if (output->numel() != 1)
            throw runtime_error("grad(): the output has " + to_string(output->numel()) +
                                " elements; pass grad_output to say how to weight them");
        grad_output = Tensor::ones(output->shape());
    }

    // order: every tensor after the tensors it was computed from
    vector<TensorPtr> order;
    unordered_set<Tensor*> seen;
    vector<pair<TensorPtr, bool>> stack{{output, false}};
    while (!stack.empty()) {
        auto [t, expanded] = stack.back();
        stack.pop_back();
        if (expanded) { order.push_back(t); continue; }
        if (!t || seen.count(t.get())) continue;
        seen.insert(t.get());
        stack.push_back({t, true});
        if (t->grad_fn())
            for (auto& in : t->grad_fn()->inputs)
                if (in && !seen.count(in.get())) stack.push_back({in, false});
    }

    // Walk backwards accumulating gradients as tensors. With create_graph the
    // ops that compute them record a graph, so the result can be
    // differentiated again; without it they are plain values.
    unordered_map<Tensor*, TensorPtr> g;
    g[output.get()] = grad_output;
    const bool previous = grad_enabled();
    grad_enabled() = create_graph;
    try {
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const TensorPtr& t = *it;
            if (!t->grad_fn()) continue;
            auto found = g.find(t.get());
            if (found == g.end()) continue;
            const auto& node = t->grad_fn();
            if (!node->backward_graph)
                throw runtime_error(string("grad(): '") + node->name +
                                    "' has no differentiable backward yet, so it cannot be on "
                                    "the path; .backward() still works through it");
            auto parts = node->backward_graph(t, found->second);
            for (size_t i = 0; i < node->inputs.size() && i < parts.size(); i++) {
                const TensorPtr& in = node->inputs[i];
                if (!parts[i] || !in || !in->requires_grad()) continue;
                auto& slot = g[in.get()];
                slot = slot ? add(slot, parts[i]) : parts[i];
            }
        }
    } catch (...) {
        grad_enabled() = previous;
        throw;
    }
    grad_enabled() = previous;

    // an input the output does not depend on gets zeros
    vector<TensorPtr> result;
    for (auto& in : inputs) {
        auto found = g.find(in.get());
        result.push_back(found != g.end() ? found->second : Tensor::zeros(in->shape()));
    }
    return result;
}

} // namespace autograd
