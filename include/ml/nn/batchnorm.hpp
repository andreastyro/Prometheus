#pragma once
#include "ml/nn/module.hpp"

// BatchNorm normalises activations across the batch dimension.
//
// For each feature (column), it computes the mean and variance across all
// samples in the batch, then normalises: (x - mean) / sqrt(var + eps).
// Learned parameters gamma (scale) and beta (shift) let the layer undo
// the normalisation if needed.
//
// Why it helps:
//   - Keeps activations in a stable range as they flow through deep networks
//   - Reduces sensitivity to weight initialisation
//   - Acts as a mild regulariser
//
// Note: behaviour differs between training and inference. During training it
// uses batch statistics; during inference it should use running averages
// (running stats not yet implemented — currently always uses batch stats).
//
// Input shape: [batch_size, num_features]
class BatchNorm : public Module {
public:
    TensorPtr gamma; // [num_features] — learned scale, initialised to 1
    TensorPtr beta;  // [num_features] — learned shift, initialised to 0
    float eps;       // small constant added to variance to avoid division by zero
    bool training;   // currently unused; reserved for running-stat mode

    BatchNorm(int num_features, float eps = 1e-5f, bool training = true);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override; // returns {gamma, beta}
};
