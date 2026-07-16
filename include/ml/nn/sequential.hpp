#pragma once
#include "ml/tensor.hpp"
#include "ml/nn/module.hpp"
#include <vector>

/// Chains multiple layers together into one model.
///
/// The output of each layer becomes the input of the next.
/// This is the standard way to build a feedforward neural network.
///
/// Example:
///   Sequential model({
///       new Linear(2, 32),
///       new ReLU(),
///       new Linear(32, 1),
///       new Sigmoid(),
///   });
///   TensorPtr out = model.forward(input);
class Sequential : public Module {
public:
    std::vector<Module*> layers; ///< Layers in order from input to output

    Sequential(std::vector<Module*> layers);

    /// Pass input through each layer in sequence and return the final output.
    TensorPtr forward(TensorPtr input) override;

    /// Collect parameters from all layers into a single flat list.
    std::vector<TensorPtr> parameters() override;
};
