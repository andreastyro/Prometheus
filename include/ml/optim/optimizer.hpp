#pragma once
#include <vector>
#include "ml/tensor.hpp"

// Optimizer is the abstract base for all optimisation algorithms.
//
// After loss.backward() fills in the .grad of every parameter tensor,
// the optimizer reads those gradients and updates the parameters to
// reduce the loss. Different optimizers use different update rules.
//
// Typical training loop:
//   loss = loss_fn(model.forward(x), y)  // 1. forward pass
//   loss.backward()                       // 2. compute gradients
//   optimizer.step()                      // 3. update weights
//   optimizer.zero_grad()                 // 4. clear gradients for next iter
class Optimizer {
public:
    std::vector<TensorPtr> parameters; // the tensors this optimizer is responsible for

    // Apply one gradient update to all parameters using the current .grad values
    virtual void step() = 0;

    // Reset all gradients to zero.
    // Must be called after each step, otherwise gradients accumulate across iterations.
    void zero_grad() {
        for (auto& p : parameters)
            p->reset_grad();
    }
};
