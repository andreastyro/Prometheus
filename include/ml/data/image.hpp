#pragma once
#include "ml/tensor.hpp"
#include <string>

// Load an image from disk and return it as a tensor.
//
// Returns shape [C, H, W]:
//   C = 1 for greyscale, 3 for RGB
//   H = image height in pixels
//   W = image width in pixels
//
// If normalize=true (default), pixel values are scaled from [0, 255] to [0, 1].
// Normalised inputs generally train faster and more stably.
//
// Typical use: feed the tensor directly into a Conv2D layer.
TensorPtr load_image(const std::string& path, bool normalize = true);
