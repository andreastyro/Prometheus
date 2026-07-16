#pragma once
#include "ml/tensor.hpp"
#include <vector>

/// Weight initialisation strategies — how to fill a layer's weights at the start.
enum class WeightInit {
    DEFAULT, ///< Small random values — a safe general choice for most layers
    XAVIER,  ///< Scales by sqrt(2 / (fan_in + fan_out)) — good for tanh/sigmoid layers
    KAIMING, ///< Scales by sqrt(2 / fan_in) — designed for ReLU layers
};

/// Base class for every layer in the library.
///
/// Every layer (Linear, ReLU, LSTM, etc.) inherits from Module.
/// A Module must be able to:
///   1. Run a forward pass — turn input into output
///   2. Report its learnable parameters — weights, biases, etc.
///
/// This common interface lets Sequential chain layers together and lets
/// optimizers collect all parameters with a single call.
class Module {
public:
    /// Compute the output of this layer given an input tensor.
    /// Every subclass must implement this.
    virtual TensorPtr forward(TensorPtr input) = 0;

    /// Return all learnable tensors in this layer (weights, biases, gamma, beta, etc.).
    /// The optimizer calls this after backward() to know what to update.
    virtual std::vector<TensorPtr> parameters() = 0;

    virtual ~Module() {}
};
