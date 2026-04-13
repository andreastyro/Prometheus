#pragma once
#include "ml/tensor.hpp"
#include <string>

std::pair<TensorPtr, TensorPtr> read_csv(const std::string& path, int y_col = -1, bool header = true);