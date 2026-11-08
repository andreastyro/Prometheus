#pragma once
#include <functional>
#include <memory>
#include "ml/tensor.hpp"

/// Element-wise addition: a + b
TensorPtr add(TensorPtr a, TensorPtr b);
/// Scalar addition: scalar + a (every element)
TensorPtr add(float scalar, TensorPtr a);

/// Element-wise subtraction: a - b
TensorPtr subtract(TensorPtr a, TensorPtr b);
/// Scalar subtraction: scalar - a
TensorPtr subtract(float scalar, TensorPtr a);
/// Scalar subtraction: a - scalar
TensorPtr subtract(TensorPtr a, float scalar);

/// Element-wise multiplication: a * b
TensorPtr multiply(TensorPtr a, TensorPtr b);
/// Scalar multiplication: scalar * a
TensorPtr multiply(float scalar, TensorPtr a);

/// Element-wise division: a / b
TensorPtr divide(TensorPtr a, TensorPtr b);
/// Scalar division: scalar / a
TensorPtr divide(float scalar, TensorPtr a);
/// Scalar division: a / scalar
TensorPtr divide(TensorPtr a, float scalar);

/// Matrix multiplication: a [m, k] @ b [k, n] -> result [m, n]
/// This is the core operation behind Linear layers and attention.
TensorPtr matmul(TensorPtr a, TensorPtr b);

/// ReLU activation: max(0, x) — zeroes out negatives, keeps positives unchanged.
TensorPtr relu(TensorPtr a);

/// Sigmoid activation: 1 / (1 + e^-x) — squashes any value into (0, 1).
TensorPtr sigmoid(TensorPtr a);

/// Tanh activation: (e^x - e^-x) / (e^x + e^-x) — squashes into (-1, 1).
TensorPtr tanh_op(TensorPtr a);

/// Softmax: converts raw scores into probabilities that sum to 1.
TensorPtr softmax(TensorPtr a);

/// Natural logarithm: ln(x)
TensorPtr log_op(TensorPtr a);

/// Exponential: e^x
TensorPtr exp_op(TensorPtr a);

/// Element-wise power: x^p
TensorPtr pow_op(TensorPtr a, float p);

/// Element-wise square root: sqrt(x)
TensorPtr sqrt_op(TensorPtr a);

/// Element-wise absolute value: |x|
TensorPtr abs_op(TensorPtr a);

/// Sum along an axis. axis=-1 reduces all elements to a scalar.
/// axis=0 sums across rows; axis=1 sums across columns.
TensorPtr sum(TensorPtr a, int dim = -1, bool keepdim = false);

/// Mean along an axis. axis=-1 averages all elements to a scalar.
TensorPtr mean(TensorPtr a, int dim = -1, bool keepdim = false);

/// Single maximum value across all elements.
TensorPtr max_op(TensorPtr a);

/// Single minimum value across all elements.
TensorPtr min_op(TensorPtr a);

/// Clamp every element to [min_val, max_val].
/// Gradient passes through where value was not clamped, zero otherwise.
TensorPtr clip(TensorPtr a, float min_val, float max_val);


/// Returns a tensor where each value is the index of the max along the last axis.
/// Used to convert softmax output into class predictions.
TensorPtr argmax(TensorPtr a);

/// GELU: Gaussian Error Linear Unit — x * Φ(x), approximated with tanh.
/// The activation used inside GPT-2/3/4 feedforward blocks.
TensorPtr gelu(TensorPtr a);



/// Concatenate tensors along `axis`. All inputs must agree on every dimension
/// except that one. Gradient-tracking: the backward splits the output gradient
/// back into the matching slice of each input.
///
/// Needed for UNet-style skip connections, which join an encoder feature map to
/// a decoder one along the channel axis. Not expressible from the other ops.
TensorPtr concat_op(std::vector<TensorPtr> inputs, int axis = 0);

/// a @ b^T, without ever building b^T.
///   a [m, k],  b [n, k]  ->  [m, n]
///
/// Equivalent to matmul(a, transpose_op(b)) but skips the copy: the transpose
/// becomes a flag to BLAS, or a different loop order in the tiled kernel.
/// Prefer this wherever a weight is stored [out, in] and applied as x @ W^T.
TensorPtr matmul_bt(TensorPtr a, TensorPtr b);

/// residual + x @ W + bias, as one op (PyTorch's addmm).
///   x [rows, in],  W [in, out],  bias [out] or null,  residual [rows, out] or null
///
/// Written as matmul -> broadcast_add -> add, this is three full passes over
/// a [rows, out] buffer and two throwaway tensors. Here the output is seeded
/// with residual + bias and the gemm accumulates onto it.
TensorPtr linear_op(TensorPtr x, TensorPtr W, TensorPtr bias = nullptr,
                    TensorPtr residual = nullptr);

// ── Views and input checking ─────────────────────────────────────────────────

/// Swap two dimensions. An O(1) view sharing storage, differentiable. The
/// default swaps the last two, which for a matrix is the usual transpose.
TensorPtr transpose(TensorPtr a, int dim0 = -2, int dim1 = -1);

/// Reorder all dimensions: result dim i is input dim dims[i]. A view.
TensorPtr permute(TensorPtr a, std::vector<int> dims);

/// Same elements, new shape; one dimension may be -1 and is inferred. A view
/// when the input is contiguous, otherwise a contiguous copy then a view.
TensorPtr reshape(TensorPtr a, ShapeArg shape);

/// The same tensor in contiguous storage: the input itself when it already
/// is, otherwise a copy that stays in the graph.
TensorPtr contiguous(TensorPtr a);

/// What every op does to its inputs before reading them through data():
/// materialise a non-contiguous view, and raise for a dtype or device the
/// kernels cannot read. Exposed for kernels outside ops.cpp.
TensorPtr checked_input(const char* op, TensorPtr t);

/// The shape two shapes broadcast to (numpy's rule), or raise.
Shape broadcast_shapes(const Shape& a, const Shape& b);

/// The integer values of an index tensor -- token ids, class labels -- for
/// any dtype, exactly. A float holding a non-integer raises.
std::vector<int64_t> index_values(TensorPtr t, const char* op);

namespace autograd {

/// Gradients of `output` with respect to `inputs`, returned as tensors --
/// torch.autograd.grad. Unlike output->backward() it leaves every tensor's
/// grad buffer alone.
///
/// With create_graph the returned gradients are themselves in the graph and
/// can be differentiated again: gradients of gradients, Hessian-vector
/// products, MAML's meta-gradient. `output` must have one element unless
/// grad_output gives the weighting. An input the output does not depend on
/// gets zeros. Every op on the path needs a differentiable backward; one
/// without it raises, naming the op.
std::vector<TensorPtr> grad(TensorPtr output, std::vector<TensorPtr> inputs,
                            bool create_graph = false, TensorPtr grad_output = nullptr,
                            bool partial = false);

/// Give a fused kernel's node a differentiable backward by recomputing its
/// forward from differentiable ops and differentiating that.
///
/// The fast kernels -- convolution, the norms, attention, the recurrent
/// layers -- compute their gradients with raw loops, which cannot be
/// differentiated again. `ref` rebuilds the same output from node.inputs (in
/// that order) out of ops that can. Only higher-order gradients use it; a
/// plain .backward() still runs the fast kernel. `partial` above is what this
/// relies on: it treats the inputs as leaves, returning the derivative with
/// respect to each direct input rather than tracing past them.
void attach_reference(GradNode& node,
                      std::function<TensorPtr(const std::vector<TensorPtr>&)> ref);

} // namespace autograd

// ── Gather and scatter by flat index ─────────────────────────────────────────
//
// out[i] = x.flat[idx[i]] and its adjoint, out.flat[idx[i]] += g[i]. Each is
// the other's backward, so both are differentiable to any order. An index
// below zero reads as 0 and scatters nowhere -- the padding of a convolution.
// They are what pooling windows, embedding lookups, convolution patches,
// slices and concatenation reduce to in the reference forwards above.

/// out (shape out_shape) gathered from x's elements at idx.
TensorPtr gather_flat(TensorPtr x, std::shared_ptr<const std::vector<int64_t>> idx,
                      ShapeArg out_shape);

/// zeros(out_shape) with g's elements added at idx.
TensorPtr scatter_add_flat(TensorPtr g, std::shared_ptr<const std::vector<int64_t>> idx,
                           ShapeArg out_shape);

// ── Dtypes: promotion, casting, comparisons, masks ───────────────────────────
//
// Every op takes every dtype. float32 runs the fast kernels; float64 computes
// in double with double gradients; float16 and bfloat16 are stored narrow and
// computed with float32 arithmetic, the result rounded back; int and bool go
// through exact integer arithmetic where the result stays integral (add,
// subtract, multiply, sums) and are promoted to float32 where it cannot (true
// division, exp, ...). The layers (conv, norms, recurrent, attention) take
// float32, float16 and bfloat16, computing in float32; float64 is for the
// tensor ops above, which is what float64 gradient checks use.

/// The dtype a binary op on a and b produces -- PyTorch's rule. Categories
/// rank bool < integer < floating, the higher category wins, and within one
/// the wider type: int32+int64 -> int64, float16+bfloat16 -> float32. A 0-d
/// tensor (a wrapped scalar) only promotes across categories: an int32
/// tensor plus a 0-d int64 stays int32.
DType result_type(const TensorPtr& a, const TensorPtr& b);
/// The promotion of two dtypes, both treated as full tensors.
DType promote_types(DType a, DType b);

/// `a` in `dtype`, the tensor itself if it already is. Differentiable between
/// floating dtypes (the gradient is cast back); out of the graph otherwise.
TensorPtr cast_op(TensorPtr a, DType dtype);

/// Element-wise comparisons, broadcasting, returning bool.
TensorPtr eq(TensorPtr a, TensorPtr b);
TensorPtr ne(TensorPtr a, TensorPtr b);
TensorPtr lt(TensorPtr a, TensorPtr b);
TensorPtr le(TensorPtr a, TensorPtr b);
TensorPtr gt(TensorPtr a, TensorPtr b);
TensorPtr ge(TensorPtr a, TensorPtr b);
/// The same against a scalar.
TensorPtr eq(TensorPtr a, double s);
TensorPtr ne(TensorPtr a, double s);
TensorPtr lt(TensorPtr a, double s);
TensorPtr le(TensorPtr a, double s);
TensorPtr gt(TensorPtr a, double s);
TensorPtr ge(TensorPtr a, double s);

/// Element-wise logic on the truth of each value (non-zero is true), bool out.
TensorPtr logical_and(TensorPtr a, TensorPtr b);
TensorPtr logical_or(TensorPtr a, TensorPtr b);
TensorPtr logical_xor(TensorPtr a, TensorPtr b);
TensorPtr logical_not(TensorPtr a);
/// True when any / every element is non-zero; a 1-element bool tensor.
TensorPtr any(TensorPtr a);
TensorPtr all(TensorPtr a);
/// isnan / isinf / isfinite, element-wise, bool out.
TensorPtr isnan_op(TensorPtr a);
TensorPtr isinf_op(TensorPtr a);
TensorPtr isfinite_op(TensorPtr a);

/// cond ? a : b element-wise, all three broadcasting. Differentiable in a and
/// b: each gets the gradient where it was chosen.
TensorPtr where(TensorPtr cond, TensorPtr a, TensorPtr b);
/// a with `value` wherever mask is true. Differentiable in a (zero gradient
/// at the filled positions). The mask broadcasts to a.
TensorPtr masked_fill(TensorPtr a, TensorPtr mask, double value);

/// Python's // and %: floor division, and the remainder taking the divisor's
/// sign. Exact on integers (dividing by zero raises); floor of the true
/// quotient on floats. Not differentiable.
TensorPtr floor_divide(TensorPtr a, TensorPtr b);
TensorPtr remainder(TensorPtr a, TensorPtr b);

/// A 0-d tensor holding one value, for the scalar side of a mixed op.
TensorPtr scalar_tensor(double value, DType dtype);

// ── Mixed precision ──────────────────────────────────────────────────────────
//
// torch.autocast: inside the scope matmul, matmul_bt and linear_op (so
// Linear) compute in the scope's dtype (float16 or bfloat16) -- inputs
// rounded to it, products accumulated in float32, the result returned in it
// -- while the ops that need range (softmax, log, exp, the reductions, the
// norms, the losses) return float32. Gradients flow back through the casts in
// float32. Pair float16 with GradScaler (ml/optim/grad_scaler.hpp), whose loss
// scaling keeps small gradients from flushing to zero.

/// The active autocast dtype, or Float32 when autocast is off.
DType autocast_dtype();
/// Set it (Float32 turns autocast off) and return the previous one. The
/// Autocast scope below is the usual way.
DType set_autocast_dtype(DType dtype);

struct Autocast {
    DType previous;
    explicit Autocast(DType dtype = DType::BFloat16);
    ~Autocast();
    Autocast(const Autocast&) = delete;
    Autocast& operator=(const Autocast&) = delete;
};

/// Inside autocast, `t` (if floating) cast to the autocast dtype; otherwise t.
TensorPtr autocast_input(TensorPtr t);
