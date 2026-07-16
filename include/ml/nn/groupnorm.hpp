#pragma once
#include "ml/nn/module.hpp"

/// Normalises within groups of channels, per sample.
///
/// Splits channels into num_groups equal groups, then normalises each group
/// independently (mean/variance computed within that group only).
///
/// Where it sits relative to other norms:
///   BatchNorm — normalises one channel across the whole batch (bad for small batches)
///   LayerNorm — normalises all channels within one sample (used in transformers)
///   GroupNorm — normalises groups of channels within one sample (used in vision models)
///
/// Works well with small batch sizes since it never looks across samples.
/// Common in object detection and image segmentation.
/// num_groups must evenly divide num_channels.
///
/// Input: [batch, num_channels, *] — any number of spatial dims after channels
class GroupNorm : public Module {
public:
    TensorPtr gamma; ///< [num_channels] — learned scale, initialised to 1
    TensorPtr beta;  ///< [num_channels] — learned shift, initialised to 0

    int num_groups;   ///< How many groups to divide channels into
    int num_channels; ///< Total channels — must be divisible by num_groups
    float eps;        ///< Small constant to avoid division by zero

    /// @param num_groups    number of groups to split channels into
    /// @param num_channels  total channels in the input (must be divisible by num_groups)
    /// @param eps           numerical stability constant (default 1e-5)
    /// Throws std::runtime_error if num_channels % num_groups != 0
    GroupNorm(int num_groups, int num_channels, float eps = 1e-5f);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override; ///< Returns {gamma, beta}
};
