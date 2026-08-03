#pragma once
#include <vector>
#include <string>
#include "ml/tensor.hpp"
#include "ml/optim/optimizer.hpp"

// AdamW — Adam with decoupled weight decay.
//
// Identical to Adam except that weight decay is applied directly to the
// weights rather than being folded into the gradient:
//
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g²
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   param -= lr * m_hat / (sqrt(v_hat) + eps)   ← same as Adam
//   param -= lr * weight_decay * param            ← decoupled weight decay
//
// Decoupled weight decay (Loshchilov & Hutter, 2017) is the standard for
// training transformers. L2 regularization through the gradient (as Adam
// does it) interacts badly with the adaptive step sizes; AdamW fixes this.
//
// Default weight_decay = 0.01 is a good starting point for transformers.
class AdamW : public Optimizer {
public:
    float lr;           // learning rate
    float beta1;        // decay rate for gradient mean
    float beta2;        // decay rate for gradient variance
    float eps;          // numerical stability (1e-8)
    float weight_decay; // strength of the weight penalty
    int   t;            // step counter for bias correction

    std::vector<std::vector<float>> m; // first moment per parameter
    std::vector<std::vector<float>> v; // second moment per parameter

    AdamW(std::vector<TensorPtr> params,
          float lr           = 0.001f,
          float beta1        = 0.9f,
          float beta2        = 0.999f,
          float eps          = 1e-8f,
          float weight_decay = 0.01f);

    void step() override;

    // Save / restore the internal optimiser state (t, m, v) to a binary file.
    // Call these alongside save_checkpoint / load_checkpoint to fully
    // checkpoint a training run so it can be resumed exactly.
    void save_state(const std::string& path) const;
    void load_state(const std::string& path);
};
