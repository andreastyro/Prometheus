#pragma once
#include "ml/nn/module.hpp"
#include <vector>

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
// Behaviour differs between training and inference:
//
//   training  — normalise with the statistics of the current batch, and fold
//               them into a running estimate
//   inference — normalise with the running estimate, so a single sample gets
//               the same treatment it would inside a batch
//
// Using batch statistics at inference time makes a prediction depend on the
// other samples that happen to share its batch, and is undefined for a batch
// of one (zero variance). Call eval() before predicting.
//
// Input shape: [batch_size, num_features]
class BatchNorm : public Module {
public:
    TensorPtr gamma; // [num_features] — learned scale, initialised to 1
    TensorPtr beta;  // [num_features] — learned shift, initialised to 0
    float eps;       // small constant added to variance to avoid division by zero
    bool training;   // true: batch statistics and update the running estimate

    float momentum;              ///< running estimate blend factor, PyTorch default 0.1
    std::vector<float> running_mean; ///< [num_features], starts at 0
    std::vector<float> running_var;  ///< [num_features], starts at 1

    /// On the GPU the running estimate lives on the device and is updated
    /// there, so a training step never waits to read it back. Call this (or
    /// move the module to the CPU) before reading the two vectors in C++;
    /// Python's bn.running_mean / running_var do it for you.
    void sync_running_stats();
    void to(Device device) override;

    BatchNorm(int num_features, float eps = 1e-5f, bool training = true,
              float momentum = 0.1f);

    /// Switch to inference: use the running estimate and stop updating it.
    void eval()  { training = false; }
    /// Switch back to training: use batch statistics and keep updating.
    void train() { training = true; }

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override; // returns {gamma, beta}

private:
    TensorPtr running_mean_dev_, running_var_dev_;   // the device copies, while on the GPU
};
