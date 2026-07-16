#pragma once
#include <vector>
#include "ml/tensor.hpp"
#include "ml/optim/optimizer.hpp"

// SGD — Stochastic Gradient Descent (with optional momentum).
//
// The simplest optimizer. Each parameter is updated by a small step
// in the direction that reduces the loss:
//   param -= lr * grad
//
// With momentum (momentum > 0), instead of using the raw gradient each step,
// it accumulates a "velocity" that builds up in consistent directions
// and dampens noisy oscillations — like a ball rolling down a hill:
//   velocity = momentum * velocity + grad
//   param    -= lr * velocity
//
// When to use:
//   Good for simple tasks and when you need interpretable behaviour.
//   Adam usually converges faster, but SGD with momentum can achieve
//   better final accuracy with careful learning rate tuning.
class SGD : public Optimizer {
public:
    float lr;       // learning rate — how large each step is
    float momentum; // 0 = no momentum (plain SGD), 0.9 is a common value

    std::vector<std::vector<float>> velocity; // one velocity vector per parameter tensor

    SGD(std::vector<TensorPtr> params, float lr, float momentum = 0.0f);

    void step() override;
};
