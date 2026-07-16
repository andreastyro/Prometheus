#pragma once
#include "ml/nn/module.hpp"

/// Randomly zeroes out neurons during training to prevent overfitting.
///
/// Each neuron is independently set to zero with probability `rate`.
/// The remaining neurons are scaled up by 1/(1-rate) so the expected
/// output stays the same. This forces each neuron to learn independently
/// rather than relying on others — a form of regularisation.
///
/// During inference (training=false) dropout is disabled and all neurons pass through.
/// Typical rate: 0.2 to 0.5 between hidden layers.
class Dropout : public Module {
public:
    float rate;    ///< Probability of zeroing a neuron — e.g. 0.3 means 30% are dropped
    bool training; ///< Set true during training (dropout active), false at inference time

    /// @param rate     drop probability in [0, 1)
    /// @param training whether dropout is active (disable for inference)
    Dropout(float rate, bool training = true);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override; ///< No learnable parameters
};
