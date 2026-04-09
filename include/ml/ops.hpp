#pragma once
#include "ml/tensor.hpp"

Tensor add(Tensor& a, Tensor& b);
Tensor multiply(Tensor& a, Tensor& b);
Tensor matmul(Tensor& a, Tensor& b);
Tensor relu(Tensor& a);
Tensor sigmoid(Tensor& a);
Tensor subtract(Tensor& a, Tensor& b);
Tensor divide(Tensor& a, Tensor& b);
Tensor tanh_op(Tensor& a);
Tensor softmax(Tensor& a);
Tensor log_op(Tensor& a);
Tensor exp_op(Tensor& a);
Tensor pow_op(Tensor& a, float p);
Tensor sqrt_op(Tensor& a);
Tensor abs_op(Tensor& a);
Tensor sum(Tensor& a, int axis = -1);
Tensor mean(Tensor& a, int axis = -1);
Tensor max_op(Tensor& a);
Tensor min_op(Tensor& a);
Tensor clip(Tensor& a, float min_val, float max_val);
Tensor broadcast_add(Tensor& a, Tensor& b);
