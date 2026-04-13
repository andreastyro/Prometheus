#pragma once
#include "ml/tensor.hpp"
#include <string>
#include <vector>

void save(const std::string& path, std::vector<TensorPtr> params);
std::vector<TensorPtr> load(const std::string& path);
