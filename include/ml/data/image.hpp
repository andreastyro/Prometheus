#pragma once
#include "ml/tensor.hpp"
#include <string>

// Loads an image from disk and returns a tensor with shape [C, W, H]
// Pixel values are normalized to [0, 1] if normalize=true
TensorPtr load_image(const std::string& path, bool normalize = true);
