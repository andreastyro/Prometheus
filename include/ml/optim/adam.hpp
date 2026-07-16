#pragma once
#include <vector>
#include "ml/tensor.hpp"
#include "ml/optim/optimizer.hpp"

// Adam — Adaptive Moment Estimation.
//
// The most widely used optimizer. It adapts the learning rate for each
// parameter individually based on how its gradient has been behaving:
//   - Parameters with large, consistent gradients get a smaller effective lr
//   - Parameters with small or noisy gradients get a larger effective lr
//
// It tracks two exponential moving averages of the gradient:
//   m = beta1 * m + (1 - beta1) * grad        (mean / direction)
//   v = beta2 * v + (1 - beta2) * grad²       (variance / magnitude)
//
// Then applies a bias correction and updates:
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   param -= lr * m_hat / (sqrt(v_hat) + eps)
//
// Default hyperparameters (lr=0.001, beta1=0.9, beta2=0.999) work well
// out of the box for most tasks. Start here before trying other optimizers.
class Adam : public Optimizer {
public:
    float lr;    // learning rate — typically 1e-3 or 1e-4
    float beta1; // decay rate for gradient mean (how quickly old gradients fade)
    float beta2; // decay rate for gradient variance (typically very close to 1)
    float eps;   // small constant to prevent division by zero (1e-8)
    int t;       // step counter used for bias correction

    std::vector<std::vector<float>> m; // first moment (mean of gradients) per parameter
    std::vector<std::vector<float>> v; // second moment (mean of squared gradients) per parameter

    Adam(std::vector<TensorPtr> params,
         float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

    void step() override;
};
