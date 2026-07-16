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
