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

    /// Default: an empty shell. Used for views, which describe elements they
    /// do not own, so allocating storage for them would defeat the purpose --
    /// constructing then clearing a 6.1M-element buffer made transpose_view
    /// only 4x faster than the copy it replaces instead of free.
    Tensor() = default;

    /// Create a tensor from existing data.
    ///
    /// `grad` is deliberately left empty. Both constructors used to size it
    /// alongside `data`, which doubled the memory of every tensor in the
    /// program -- including constants, inputs and the intermediate activations
    /// of an inference pass, none of which ever receive a gradient. It is
    /// allocated on first need instead, by ensure_grad().
    Tensor(std::vector<int> shape_, std::vector<float> data_) {
        shape = shape_;
        data  = data_;
    }

    /// Create a zero-filled tensor with the given shape.
    Tensor(std::vector<int> shape_) {
        shape = shape_;
        int total = 1;
        for (int i : shape)
            total *= i;
        data.resize(total, 0.0f);
    }

    // ── Views ────────────────────────────────────────────────────────────
    //
    // A view shares another tensor's storage and describes a different way of
    // walking it. strides[d] is how many elements to step to advance one along
    // dimension d, so a transpose is a swap of two strides and costs nothing,
    // where copying a 16000x384 table costs 24.6 MB.
    //
    // THE GRADIENT CONVENTION, which is the part that is easy to get wrong:
    // a view's `grad` is row-major over the VIEW's own shape, exactly as it
    // would be for an owned tensor of that shape. It is not indexed through
    // the strides. Every backward_fn in the library writes `grad[i]` in
    // row-major order, so this is the only convention that lets views and
    // owned tensors share that code. A first attempt at this indexed a view's
    // gradient by storage offset in one place and row-major in another, which
    // disagreed silently wherever it did not crash.
    //
    // Views own no `data`, so an op that reads `data` directly must take
    // contiguous() first. Ops that can walk strides consume the view as is.
    std::shared_ptr<Tensor> base;   ///< storage owner, null when this owns its data
    std::vector<int> strides;       ///< element stride per dimension
    int offset = 0;                 ///< index of the first element in base->data

    bool is_view() const { return base != nullptr; }

    /// Row-major strides for a shape, which is what an owned tensor has.
    static std::vector<int> contiguous_strides(const std::vector<int>& shape) {
        std::vector<int> st(shape.size(), 1);
        for (int d = (int)shape.size() - 2; d >= 0; d--)
            st[d] = st[d + 1] * shape[d + 1];
        return st;
    }

    /// True when the elements can be read straight out of storage in order.
    bool is_contiguous() const {
        if (!is_view()) return true;
        return offset == 0 && strides == contiguous_strides(shape);
    }

    /// Element at a multi-dimensional index, honouring strides.
    float view_at(const std::vector<int>& idx) const {
        int flat = offset;
        for (std::size_t d = 0; d < idx.size(); d++)
            flat += idx[d] * strides[d];
        return is_view() ? base->data[flat] : data[flat];
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
        // num_el() rather than data.size(): they agree for an owned tensor,
        // but sizing from the element count is the correct rule and is what a
        // tensor that does not own its storage would need.
        grad.assign((std::size_t)num_el(), 0.0f);
    }

    /// Allocate the gradient buffer if it has not been allocated yet.
    ///
    /// Every backward_fn writes with `grad[i] +=`, so the buffer has to exist
    /// and be zeroed before the first write. make_node() calls this on each
    /// input, which covers everything that takes part in a graph; anything
    /// else never needs one.
    void ensure_grad() {
        if (grad.size() != (std::size_t)num_el())
            grad.assign((std::size_t)num_el(), 0.0f);
    }

    /// True once a gradient buffer exists. Useful for asserting that a tensor
    /// actually took part in a backward pass.
    bool has_grad() const { return grad.size() == (std::size_t)num_el(); }

    /// Create a tensor filled with random values from N(0, 1).
    /// Used to initialise weights — random breaks symmetry so neurons learn different things.
    static TensorPtr randn(std::vector<int> shape);

    /// Seed the library-wide RNG so randn() (and therefore weight init)
    /// is reproducible across runs.
    static void manual_seed(unsigned int seed);

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

/// The library-wide RNG, seeded by Tensor::manual_seed. Layers that need
/// randomness (Dropout) draw from this so a run can be reproduced.
std::mt19937& prometheus_rng();

/// Whether ops should record a computation graph. True by default.
///
/// At inference there is nothing to backpropagate, so building the graph is
/// pure waste: every op would allocate a GradNode holding shared_ptrs to its
/// inputs and a std::function closure, and the whole graph would be kept alive
/// until the result went out of scope. Turning this off skips all of it.
bool &grad_enabled();

/// Scoped guard. Restores the previous setting on destruction, so it nests and
/// is exception-safe:
///
///     {
///         NoGrad _;
///         auto logits = model.forward(x);   // no graph built
///     }
struct NoGrad {
    bool previous;
    NoGrad() : previous(grad_enabled()) { grad_enabled() = false; }
    ~NoGrad() { grad_enabled() = previous; }
};

/// Wire a new result tensor into the computation graph.
/// Sets requires_grad=true on result, creates a GradNode with the given inputs,
/// and returns the node so the caller can attach a backward_fn.
inline std::shared_ptr<GradNode> make_node(TensorPtr result, std::vector<TensorPtr> inputs) {
    auto node = std::make_shared<GradNode>();
    node->inputs = inputs;

    // Anything joining the graph will be written to by a backward_fn, so give
    // it a gradient buffer now. Done before the NoGrad check below is pointless
    // -- under NoGrad nothing will ever run backward -- so it comes after.

    // Under NoGrad the node is returned but not attached: the caller still
    // assigns its backward_fn, which is then discarded with the node, and the
    // result stays a leaf. That keeps every call site unchanged while making
    // inference allocate nothing beyond the output tensor itself.
    if (!grad_enabled())
        return node;

    for (auto& in : inputs)
        if (in && in->requires_grad)
            in->ensure_grad();
    result->ensure_grad();

    result->requires_grad = true;
    result->grad_fn = node;
    return node;
}
