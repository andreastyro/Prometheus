#pragma once
#include "ml/tensor.hpp"

Tensor mse_loss(Tensor& pred, Tensor& target);
Tensor mae_loss(Tensor& pred, Tensor& target);
Tensor bce_loss(Tensor& pred, Tensor& target);
Tensor cross_entropy_loss(Tensor& pred, Tensor& target);