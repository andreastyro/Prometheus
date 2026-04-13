#pragma once
#include "ml/tensor.hpp"

float accuracy(TensorPtr pred, TensorPtr target);
float precision(TensorPtr pred, TensorPtr target);
float recall(TensorPtr pred, TensorPtr target);
float f1_score(TensorPtr pred, TensorPtr target);
TensorPtr confusion_matrix(TensorPtr pred, TensorPtr target);
float r2_score(TensorPtr pred, TensorPtr target);
