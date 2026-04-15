#pragma once
#include <vector>
#include <functional>
#include <memory>

// Forward declare Tensor so GradNode can reference it
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

struct GradNode {
    // The tensors that were used as inputs to create this tensor
    std::vector<TensorPtr> inputs;

    // The function that computes and accumulates gradients into inputs
    std::function<void()> backward_fn;
};

