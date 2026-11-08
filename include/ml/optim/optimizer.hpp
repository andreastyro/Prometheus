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
/// A per-parameter optimizer buffer (momentum, Adam's moments) on the GPU,
/// made from the host copy the first time its parameter is updated there --
/// so an optimizer built before model.to(cuda) works the same as one built
/// after. to_host() copies back, for save_state; reset() drops the device
/// copies, after load_state.
struct DeviceBuffers {
    mutable std::vector<std::shared_ptr<Storage>> dev;
    float* get(std::size_t i, const std::vector<float>& host);
    void   to_host(std::size_t i, std::vector<float>& host) const;
    void   reset() { dev.clear(); }
};

/// The device pointers of a GPU parameter and its gradient (null if it has none).
float*       device_weights(const TensorPtr& p);
const float* device_grad(const TensorPtr& p);

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
