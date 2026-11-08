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
    /// Propagate this node's output gradient into its inputs' gradients.
    ///
    /// The output tensor is passed in rather than captured. A closure that
    /// captured its own output made a cycle -- output -> grad_fn -> node ->
    /// closure -> output -- that reference counting can never free, so every
    /// graph ever built leaked: measured at 6.3 MB per training step on a
    /// small model, growing without bound. With the output as an argument,
    /// nothing in the graph points back towards the loss, and the whole graph
    /// goes when the last reference to the output does.
    std::function<void(const TensorPtr& output)> backward_fn;

    /// The op's name, for error messages.
    const char* name = "op";

    /// The same backward written with differentiable ops: given the output
    /// and dL/d(output) as tensors, return dL/d(input) for each input, in
    /// the order of `inputs` (nullptr where an input needs none).
    ///
    /// backward_fn writes raw float buffers, which is fast but is a dead end:
    /// nothing records how those numbers were made, so they cannot be
    /// differentiated again. This version builds graph like any other op
    /// when gradients are enabled, which is what autograd::grad with
    /// create_graph relies on for gradients of gradients. An op without one
    /// cannot be differentiated through by autograd::grad, which says so.
    std::function<std::vector<TensorPtr>(const TensorPtr& output,
                                         const TensorPtr& grad_output)> backward_graph;
};
