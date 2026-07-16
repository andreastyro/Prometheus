#pragma once
#include "ml/nn/sequential.hpp"
#include "ml/nn/activations.hpp"
#include "ml/nn/linear.hpp"
#include "ml/nn/dropout.hpp"
#include <string>

// Print a human-readable summary of the model to stdout.
//
// For each layer shows:
//   - Layer type (Linear, ReLU, Dropout, etc.)
//   - Parameter shapes (e.g. weights [64, 32], bias [32])
//   - Number of trainable parameters in that layer
// Followed by the total parameter count across the whole model.
//
// Useful for quickly checking that your architecture is what you intended
// and estimating model size before training.
//
// Usage:
//   model_summary({new Linear(2, 32), new ReLU(), new Linear(32, 1)});
void model_summary(std::vector<Module*> layers);
