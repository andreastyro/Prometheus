#pragma once
#include "ml/nn/module.hpp"

/// Normalises across features within a single sample (per row / per token).
///
/// For each row, computes mean and variance across all features in that row, then:
///   output = gamma * (x - mean) / sqrt(var + eps) + beta
///
/// Contrast with BatchNorm:
///   BatchNorm — "centre this one feature across all samples in the batch"
///   LayerNorm — "centre all features within this one sample"
///
/// LayerNorm is the standard choice in transformers — it works correctly even
/// with batch_size=1 and never depends on other samples in the batch.
///
/// gamma and beta are learned so the network can undo the normalisation if needed.
///
/// Input: [..., normalized_shape] — works on any number of leading dimensions
class LayerNorm : public Module {
public:
    TensorPtr gamma; ///< [normalized_shape] — learned scale, initialised to 1
    TensorPtr beta;  ///< [normalized_shape] — learned shift, initialised to 0

    int normalized_shape; ///< Size of the last dimension to normalise over
    float eps;            ///< Small constant added to variance to avoid division by zero

    /// @param normalized_shape  size of the last dimension (number of features per token)
    /// @param eps               numerical stability constant (default 1e-5)
    LayerNorm(int normalized_shape, float eps = 1e-5f);

    /// Normalise each row independently over its last dimension.
    TensorPtr forward(TensorPtr input) override;

    std::vector<TensorPtr> parameters() override; ///< Returns {gamma, beta}
};
