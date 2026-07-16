#pragma once
#include "ml/nn/module.hpp"

/// Fully connected layer — the most fundamental building block.
///
/// Every neuron in this layer is connected to every neuron in the previous layer.
/// Computes: output = input @ weights + bias
///
/// Example:
///   Linear(2, 8)  — takes 2 input features, produces 8 output features
///   input:  [batch, 2]
///   output: [batch, 8]
class Linear : public Module {
public:
    TensorPtr weights; ///< [in_features, out_features] — one weight per (input, output) pair
    TensorPtr bias;    ///< [out_features] — one bias per output neuron

    /// @param in_features  size of each input vector
    /// @param out_features number of neurons (size of output vector)
    /// @param init         weight init strategy: "default", "xavier", or "kaiming"
    Linear(int in_features, int out_features, std::string init = "default");

    /// Compute output = input @ weights + bias
    TensorPtr forward(TensorPtr input) override;

    /// Returns {weights, bias}
    std::vector<TensorPtr> parameters() override;
};
