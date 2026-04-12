#pragma once
#include "ml/tensor.hpp"

TensorPtr mse_loss(TensorPtr pred, TensorPtr target);
TensorPtr mae_loss(TensorPtr pred, TensorPtr target);
TensorPtr bce_loss(TensorPtr pred, TensorPtr target);
TensorPtr cross_entropy_loss(TensorPtr pred, TensorPtr target);
