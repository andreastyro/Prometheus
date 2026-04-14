#pragma once
#include "ml/tensor.hpp"

TensorPtr add(TensorPtr a, TensorPtr b);
TensorPtr add(float scalar, TensorPtr a);
TensorPtr multiply(TensorPtr a, TensorPtr b);
TensorPtr multiply(float scalar, TensorPtr a);
TensorPtr matmul(TensorPtr a, TensorPtr b);
TensorPtr relu(TensorPtr a);
TensorPtr sigmoid(TensorPtr a);
TensorPtr subtract(TensorPtr a, TensorPtr b);
TensorPtr subtract(float scalar, TensorPtr a);
TensorPtr subtract(TensorPtr a, float scalar);
TensorPtr divide(TensorPtr a, TensorPtr b);
TensorPtr divide(float scalar, TensorPtr a);
TensorPtr divide(TensorPtr a, float scalar);
TensorPtr tanh_op(TensorPtr a);
TensorPtr softmax(TensorPtr a);
TensorPtr log_op(TensorPtr a);
TensorPtr exp_op(TensorPtr a);
TensorPtr pow_op(TensorPtr a, float p);
TensorPtr sqrt_op(TensorPtr a);
TensorPtr abs_op(TensorPtr a);
TensorPtr sum(TensorPtr a, int axis = -1);
TensorPtr mean(TensorPtr a, int axis = -1);
TensorPtr max_op(TensorPtr a);
TensorPtr min_op(TensorPtr a);
TensorPtr clip(TensorPtr a, float min_val, float max_val);
TensorPtr broadcast_add(TensorPtr a, TensorPtr b);
TensorPtr argmax(TensorPtr a);
