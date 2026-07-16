#pragma once
#include "ml/nn/module.hpp"

/// Collapses all dimensions after the batch into a single vector.
///
/// Used to bridge convolutional layers (which output 3D feature maps) and
/// Linear layers (which expect a flat 1D input per sample).
/// No learnable parameters — just a reshape operation.
///
/// Example:
///   input:  [batch, 32, 7, 7]  (32 channels, 7x7 spatial)
///   output: [batch, 1568]      (32 * 7 * 7 = 1568)
class Flatten : public Module {
public:
    Flatten() = default;

    /// Reshape input from [batch, C, H, W] to [batch, C*H*W]
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override; ///< No learnable parameters
};
