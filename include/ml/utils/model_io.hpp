#pragma once
#include "ml/tensor.hpp"
#include <string>
#include <vector>

// Save model parameters to a binary file.
//
// Writes each tensor's shape and data sequentially. Use this to checkpoint
// a trained model so you can reload it later without retraining.
//
// Usage:
//   save("model.bin", model.parameters());
void save(const std::string& path, std::vector<TensorPtr> params);

// Load model parameters from a file saved with save().
//
// Returns the tensors in the same order they were saved.
// Assign them back to your model's parameters to restore weights.
//
// Usage:
//   auto params = load("model.bin");
std::vector<TensorPtr> load(const std::string& path);
