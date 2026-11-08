#pragma once
// Helpers shared by ops.cpp and dtype_ops.cpp. Not installed.
#include "ml/ops.hpp"
#include "ml/tensor.hpp"
#include <cstdint>
#include <functional>
#include <string>

std::string shape_str(const Shape& s);

/// Strides that read a contiguous tensor of shape `in` at the positions of
/// `out`: its own row-major strides, and 0 wherever it is being broadcast.
Shape broadcast_strides(const Shape& in, const Shape& out);

/// Calls f(i, ia, ib) for every output element: i its row-major index, ia and
/// ib the matching elements of the two inputs. Rows (everything but the last
/// dimension) are independent and run in parallel when `parallel` is set; a
/// caller that scatters into a broadcast input must not set it.
template <class F>
void broadcast_walk(const Shape& out, const Shape& sa, const Shape& sb, bool parallel, F f) {
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

/// Sum a gradient back down to `shape`, undoing a broadcast.
TensorPtr sum_to(TensorPtr g, const Shape& shape);
/// Broadcast a gradient up to `shape`: the backward of a reduction.
TensorPtr expand_to(TensorPtr g, const Shape& shape);

// ── Routing an op by dtype ───────────────────────────────────────────────────
//
// Each op calls one of these first. They return nullptr for float32 inputs,
// and the op's own float32 kernel runs; otherwise they return the result:
//
//   float16 / bfloat16  cast to float32, the op itself, cast back
//   float64             a double kernel here, differentiated through
//                       `graph` -- the op's gradient written with ops
//   int / bool          exact integer arithmetic where the result stays
//                       integral, else promoted to float32
namespace dispatch {

enum class Bin { Add, Sub, Mul, Div };

/// dL/da given a, the op's output and dL/d(out), as ops.
using Graph1 = std::function<TensorPtr(const TensorPtr& a, const TensorPtr& out, const TensorPtr& G)>;
using Self1  = std::function<TensorPtr(TensorPtr)>;
using Self2  = std::function<TensorPtr(TensorPtr, TensorPtr)>;

/// Element-wise f. `keeps_int`: an integer input gives an integer result
/// (relu, abs, x - 1) rather than float32 (exp, sigmoid). `range_op`: under
/// autocast a half input returns float32 (exp, log, pow, softmax).
TensorPtr unary(const char* name, const TensorPtr& a, std::function<double(double)> f,
                Graph1 graph, const Self1& self, bool keeps_int = false, bool range_op = false,
                const char* cuda_kernel = nullptr, float p0 = 0.0f, float p1 = 0.0f);

/// a OP b with broadcasting.
TensorPtr binary(const char* name, const TensorPtr& a, const TensorPtr& b, Bin op, const Self2& self);

/// sum or mean over `dim` (-1: everything).
TensorPtr reduce(const char* name, const TensorPtr& a, int dim, bool keepdim, bool mean,
                 const std::function<TensorPtr(TensorPtr)>& self);
TensorPtr extreme(const char* name, const TensorPtr& a, bool is_max, const Self1& self);
TensorPtr argmax(const TensorPtr& a);
TensorPtr softmax(const TensorPtr& a, const Self1& self);
TensorPtr matmul(TensorPtr a, TensorPtr b, const Self2& self);
TensorPtr concat(std::vector<TensorPtr>& inputs, int axis,
                 const std::function<TensorPtr(std::vector<TensorPtr>, int)>& self);
TensorPtr gather(const TensorPtr& x, std::shared_ptr<const std::vector<int64_t>> idx,
                 const Shape& out_shape);
TensorPtr scatter_add(const TensorPtr& g, std::shared_ptr<const std::vector<int64_t>> idx,
                      const Shape& out_shape);

} // namespace dispatch

// GPU kernels (src/cuda/cuda_ops.cpp). Each attaches a backward_graph.
namespace cuda_ops {
/// binary op: 0 add, 1 subtract, 2 multiply, 3 divide
TensorPtr binary(int op, const char* name, const TensorPtr& a, const TensorPtr& b);
/// element-wise `kernel` (u_exp, u_smul, ...) with two scalar parameters
TensorPtr unary(const char* kernel, const char* name, const TensorPtr& a, float p0, float p1,
                dispatch::Graph1 graph);
/// op: 0 eq 1 ne 2 lt 3 le 4 gt 5 ge 6 and 7 or 8 xor; bool out
TensorPtr compare(int op, const TensorPtr& a, const TensorPtr& b);
TensorPtr where(const TensorPtr& cond, const TensorPtr& a, const TensorPtr& b);
TensorPtr reduce(const char* name, const TensorPtr& a, int dim, bool keepdim, bool mean);
TensorPtr softmax(const TensorPtr& a);
/// a in dtype d, contiguous (no graph)
TensorPtr cast(const TensorPtr& a, DType d);
/// a contiguous copy (no graph)
TensorPtr copy(const TensorPtr& a);
/// a CPU scalar moved to the device; a GPU tensor as is
TensorPtr on_device(const TensorPtr& t, const char* op);
/// float32 grad[0..n) += g (any float dtype, any layout, on either device)
void accumulate(void* grad, int64_t n, const TensorPtr& g);
void fill(void* dst, int64_t n, float value);
TensorPtr extreme(const char* name, const TensorPtr& a, bool is_max);
TensorPtr argmax(const TensorPtr& a);
TensorPtr gather(const TensorPtr& x, const std::shared_ptr<const std::vector<int64_t>>& idx, const Shape& out_shape);
TensorPtr scatter_add(const TensorPtr& g, const std::shared_ptr<const std::vector<int64_t>>& idx, const Shape& out_shape);
TensorPtr concat(const std::vector<TensorPtr>& inputs, int axis);
/// cuBLAS; batch dims broadcast as on the CPU
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
/// mean over rows of -log softmax(logits)[target]
TensorPtr cross_entropy(const TensorPtr& logits, const std::vector<int>& targets);
/// LayerNorm forward over the last dim, one fused kernel (no graph)
TensorPtr layernorm(const TensorPtr& x, const TensorPtr& gamma, const TensorPtr& beta, float eps);
/// float32 mask of `shape`: 0 with probability `rate`, else 1/(1-rate). Philox, keyed by `seed`.
TensorPtr dropout_mask(const Shape& shape, uint64_t seed, float rate);
// optimizer updates on device buffers; v may be null for plain SGD
void sgd_step(float* w, const float* g, float* v, int64_t n, float lr, float mu);
void adam_step(float* w, const float* g, float* m, float* v, int64_t n, float lr, float b1, float b2,
               float bc1, float bc2, float eps, float decay);
void rmsprop_step(float* w, const float* g, float* v, int64_t n, float lr, float b, float eps);
double sum_squares(const float* a, int64_t n);
void scale_inplace(float* a, int64_t n, float c);
/// g /= scale in place, zeroing non-finite entries; false if there were any
bool unscale_inplace(float* g, int64_t n, float inv);
// fused backward kernels (used when the gradient is not itself recorded)
std::vector<TensorPtr> layernorm_backward(const TensorPtr& x, const TensorPtr& gamma, const TensorPtr& G, float eps);
TensorPtr gelu_backward(const TensorPtr& x, const TensorPtr& G);
/// rows of W picked by ids, as `out_shape`; backward is a scatter-add kernel
TensorPtr embedding(const TensorPtr& W, const std::vector<int64_t>& ids, const Shape& out_shape);
/// softmax(Q K^T / sqrt(D) + causal mask) V per head; q, k, v and the result [B, S, E]
TensorPtr attention(const TensorPtr& q, const TensorPtr& k, const TensorPtr& v, int heads, bool causal);
}
