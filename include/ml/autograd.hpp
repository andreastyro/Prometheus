#pragma once
#include <vector>
#include <functional>
#include <memory>

// Forward declare Tensor so GradNode can reference it without a circular include
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

// GradNode is the building block of the computation graph.
//
// Every time you call an op (add, matmul, relu, etc.) on tensors that require
// gradients, the op creates a GradNode and attaches it to the result tensor.
// The node remembers:
//   - which tensors were used as inputs
//   - how to compute and accumulate the gradient back into those inputs
//
// When you call tensor.backward(), it walks the graph in reverse (from loss
// back to weights) and calls each node's backward_fn in topological order.
struct GradNode {
    // The tensors that were fed into the op that created this node
    std::vector<TensorPtr> inputs;

    // The gradient function — when called, it reads result->grad (the incoming
    // gradient from the layer above) and accumulates the correct gradient into
    // each input tensor's .grad field
    std::function<void()> backward_fn;
};
