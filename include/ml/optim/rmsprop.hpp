#pragma once
#include "ml/tensor.hpp"
#include "ml/optim/optimizer.hpp"
#include <vector>

// RMSprop — Root Mean Square Propagation.
//
// Adapts the learning rate per parameter by dividing by a running average
// of recent squared gradients. This dampens parameters that have been
// receiving large updates and amplifies those that have been receiving small ones.
//
//   v = beta * v + (1 - beta) * grad²
//   param -= lr * grad / (sqrt(v) + eps)
//
// Particularly well suited for RNNs and non-stationary problems
// (where the gradient distribution shifts as training progresses).
// Invented by Hinton as an informal improvement over Adagrad.
class RMSprop : public Optimizer {
public:
    float lr;   // learning rate
    float beta; // decay rate for the running average of squared gradients (typically 0.9)
    float eps;  // small constant to prevent division by zero

    std::vector<std::vector<float>> v; // running average of squared gradients per parameter

    RMSprop(std::vector<TensorPtr> params, float lr = 0.001f, float beta = 0.9f, float eps = 1e-8f);

    void step() override;
};
