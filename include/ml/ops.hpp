#pragma once
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
TensorPtr sum(TensorPtr a, int axis = -1);

/// Mean along an axis. axis=-1 averages all elements to a scalar.
TensorPtr mean(TensorPtr a, int axis = -1);

/// Single maximum value across all elements.
TensorPtr max_op(TensorPtr a);

/// Single minimum value across all elements.
TensorPtr min_op(TensorPtr a);

/// Clamp every element to [min_val, max_val].
/// Gradient passes through where value was not clamped, zero otherwise.
TensorPtr clip(TensorPtr a, float min_val, float max_val);

/// Add a 1D bias to every row of a 2D tensor.
/// a: [batch, features]  b: [features]  ->  [batch, features]
TensorPtr broadcast_add(TensorPtr a, TensorPtr b);

/// Returns a tensor where each value is the index of the max along the last axis.
/// Used to convert softmax output into class predictions.
TensorPtr argmax(TensorPtr a);

/// GELU: Gaussian Error Linear Unit — x * Φ(x), approximated with tanh.
/// The activation used inside GPT-2/3/4 feedforward blocks.
TensorPtr gelu(TensorPtr a);

/// Gradient-tracking reshape. Returns a new tensor with the given shape
/// but the same element count; gradients flow back through the reshape.
TensorPtr reshape_op(TensorPtr a, std::vector<int> new_shape);

/// Gradient-tracking 2D transpose. Returns a new [cols, rows] tensor.
/// Tensor::transpose() is the non-differentiable equivalent: it detaches,
/// so it cannot be used inside a forward pass you intend to backprop through.
TensorPtr transpose_op(TensorPtr a);

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

/// O(1) transpose: a 2-D view with the strides swapped, sharing storage.
/// Nothing is copied. The result is non-contiguous, so pass it through
/// contiguous() before an op that reads `data` directly. To apply a weight
/// transposed, prefer matmul_bt, which needs neither.
TensorPtr transpose_view(TensorPtr a);

/// O(1) reshape: a view with the same element order and a new shape. A
/// reshape of contiguous storage stays contiguous.
TensorPtr reshape_view(TensorPtr a, std::vector<int> new_shape);

/// Materialise a view into owned, contiguous storage. Returns the input
/// unchanged when it already owns its data, so calling it is free in the
/// common case.
TensorPtr contiguous(TensorPtr a);
