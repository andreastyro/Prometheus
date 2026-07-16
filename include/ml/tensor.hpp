#pragma once
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>
#include "ml/autograd.hpp"

/// Core data structure — a flat array of floats with a shape attached.
///
/// Example shapes:
///   [4]           — a vector of 4 values
///   [3, 2]        — a 3x2 matrix (3 rows, 2 columns)
///   [8, 3, 32, 32] — a batch of 8 RGB images, 32x32 pixels
///
/// Tensors also carry gradient information. When requires_grad is true, every
/// op that uses this tensor attaches a GradNode to the result so that
/// backward() can propagate gradients back through the whole computation graph.
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<float> data;  ///< Flat storage — all values in row-major order
    std::vector<int>   shape; ///< Size of each dimension, e.g. {3, 2} for a 3x2 matrix
    std::vector<float> grad;  ///< Gradient of the loss w.r.t. each element in data
    bool requires_grad = false; ///< When true, ops on this tensor build the computation graph

    /// Points to the op that produced this tensor.
    /// nullptr for leaf tensors (inputs and weights). Used by backward() to walk the graph.
    std::shared_ptr<GradNode> grad_fn = nullptr;

    /// Create a tensor from existing data.
    Tensor(std::vector<int> shape_, std::vector<float> data_) {
        shape = shape_;
        data  = data_;
        grad.resize(data.size(), 0.0f);
    }

    /// Create a zero-filled tensor with the given shape.
    Tensor(std::vector<int> shape_) {
        shape = shape_;
        int total = 1;
        for (int i : shape)
            total *= i;
        data.resize(total, 0.0f);
        grad.resize(total, 0.0f);
    }

    /// Total number of elements — product of all dimension sizes.
    /// Example: shape {3, 2} -> 6 elements.
    int num_el() const {
        int total = 1;
        for (int d : shape)
            total *= d;
        return total;
    }

    /// Read the element at (row, col) — 2D tensors only.
    float get(int row, int col) {
        return data[row * shape[1] + col];
    }

    /// Write a value at (row, col) — 2D tensors only.
    void set(int row, int col, float value) {
        data[row * shape[1] + col] = value;
    }

    /// Print all values to stdout, laid out according to the tensor's shape.
    void print() const;

    /// Set every element to the given value.
    void fill(float val) {
        for (float& i : data)
            i = val;
    }

    /// Create a tensor of all zeros with the given shape.
    static TensorPtr zeros(std::vector<int> shape) {
        return std::make_shared<Tensor>(shape);
    }

    /// Create a tensor of all ones with the given shape.
    static TensorPtr ones(std::vector<int> shape) {
        auto t = std::make_shared<Tensor>(shape);
        t->fill(1.0f);
        return t;
    }

    /// Zero out all gradients. Call this before each forward pass.
    void reset_grad() {
        grad.assign(data.size(), 0.0f);
    }

    /// Create a tensor filled with random values from N(0, 1).
    /// Used to initialise weights — random breaks symmetry so neurons learn different things.
    static TensorPtr randn(std::vector<int> shape);

    /// Swap rows and columns — shape {3, 2} becomes {2, 3}. 2D only.
    TensorPtr transpose() const;

    /// Change the shape without changing the data.
    /// Total element count must stay the same — e.g. {4, 3} can become {12} or {2, 6}.
    TensorPtr reshape(std::vector<int> new_shape) const;

    /// Walk the computation graph backwards and accumulate gradients.
    /// Call this on the loss tensor after the forward pass.
    void backward();

    /// Return a copy not connected to the computation graph.
    /// Useful for inspecting values mid-graph without affecting gradients.
    TensorPtr detach() const;

    /// Convert to simulated float16 (loses precision, smaller range).
    TensorPtr half() const;

    /// Convert back from simulated float16 to float32.
    TensorPtr to_float() const;
};

/// Wire a new result tensor into the computation graph.
/// Sets requires_grad=true on result, creates a GradNode with the given inputs,
/// and returns the node so the caller can attach a backward_fn.
inline std::shared_ptr<GradNode> make_node(TensorPtr result, std::vector<TensorPtr> inputs) {
    result->requires_grad = true;
    auto node = std::make_shared<GradNode>();
    node->inputs = inputs;
    result->grad_fn = node;
    return node;
}
