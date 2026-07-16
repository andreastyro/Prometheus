#pragma once
#include "ml/nn/module.hpp"

/// Activation layers apply a non-linear function element-wise.
/// Without activations, a deep network collapses to a single linear transformation.
/// Non-linearities let the network learn curves, spirals, and complex decision boundaries.

/// ReLU — Rectified Linear Unit: f(x) = max(0, x)
/// Zeroes out negatives, keeps positives unchanged.
/// Fast and simple — the default choice for hidden layers.
class ReLU : public Module {
public:
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override; ///< No learnable parameters
};

/// Sigmoid: f(x) = 1 / (1 + e^-x)
/// Squashes any value into (0, 1).
/// Use as the final layer for binary classification — output is a probability.
class Sigmoid : public Module {
public:
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};

/// Tanh: f(x) = (e^x - e^-x) / (e^x + e^-x)
/// Squashes any value into (-1, 1). Zero-centred version of Sigmoid.
/// Often preferred over Sigmoid in RNN hidden states.
class Tanh : public Module {
public:
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};

/// Softmax: converts a vector of raw scores into probabilities that sum to 1.
/// Use as the final layer for multi-class classification.
/// Example: [2.0, 1.0, 0.5] -> [0.62, 0.23, 0.15]
class Softmax : public Module {
public:
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};
