#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "ml/autograd.hpp"

// ── dtype and device ─────────────────────────────────────────────────────────

/// The number format of every element in a tensor.
///
/// Storage and conversion work for all of them. Compute kernels are float32;
/// an op handed another dtype raises and says to convert with .to(), rather
/// than reading the bytes as floats. Integer dtypes hold token ids and
/// indices exactly, where a float32 loses precision past 2^24.
enum class DType : std::uint8_t {
    Float32, Float64, Float16, BFloat16, Int64, Int32, Bool
};

/// Where a tensor's bytes live. Only the CPU exists in this build; the field
/// is here so that adding a GPU means adding a device, not redesigning Tensor.
enum class Device : std::uint8_t { CPU, CUDA };

std::size_t dtype_size(DType d);
const char* dtype_name(DType d);
const char* device_name(Device d);
bool        is_floating(DType d);

/// Sizes and strides are 64-bit: an int overflows past 2^31 elements.
using Shape = std::vector<std::int64_t>;

/// Accepts a shape written any of the ways code already writes one:
/// {2, 3}, a std::vector<int>, or a Shape.
class ShapeArg {
public:
    ShapeArg(std::initializer_list<std::int64_t> s) : v_(s) {}
    ShapeArg(const Shape& s) : v_(s) {}
    ShapeArg(Shape&& s) : v_(std::move(s)) {}
    ShapeArg(const std::vector<int>& s) : v_(s.begin(), s.end()) {}
    const Shape& get() const { return v_; }
    operator const Shape&() const { return v_; }
private:
    Shape v_;
};

/// Caching allocator for tensor memory.
///
/// A new 4 MB buffer from the OS costs about 0.5 ms on Windows -- not to zero
/// (0.03 ms) but to page-fault in, since large blocks go straight back to the
/// OS when freed. Every op makes an output and a gradient buffer, so on a
/// [16, 64, 32, 32] activation that was the entire cost of a relu, and made
/// BatchNorm 7x slower than PyTorch. Freed buffers are kept, by exact size,
/// and handed to the next request of that size with their pages still mapped:
/// what PyTorch's caching allocator does. Small blocks bypass it; the heap
/// already recycles those cheaply.
namespace mem {
    /// At least `nbytes`, 64-byte aligned, contents unspecified.
    void* alloc(std::size_t nbytes);
    void  free(void* ptr, std::size_t nbytes) noexcept;
    /// Bytes held idle, ready for reuse.
    std::size_t cached_bytes();
    /// Return every idle buffer to the OS.
    void empty_cache();
}

/// A float buffer from the caching allocator: the gradient storage.
class FloatBuffer {
public:
    FloatBuffer() = default;
    ~FloatBuffer() { reset(); }
    FloatBuffer(const FloatBuffer&) = delete;
    FloatBuffer& operator=(const FloatBuffer&) = delete;
    float*       data()       { return p_; }
    const float* data() const { return p_; }
    std::size_t  size()  const { return n_; }
    bool         empty() const { return n_ == 0; }
    /// n elements, every one set to v.
    void assign(std::size_t n, float v);
    void reset();
private:
    float*      p_ = nullptr;
    std::size_t n_ = 0;
};

/// The bytes behind one or more tensors. A view and the tensor it was taken
/// from share one Storage; nothing else does.
class Storage {
public:
    /// `zero` false leaves the bytes as the allocator returned them, for an
    /// output whose every element the caller is about to write.
    Storage(std::size_t nbytes, DType dtype, Device device, bool zero = true);
    ~Storage();
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    void*       data()         { return ptr_; }
    const void* data()   const { return ptr_; }
    std::size_t nbytes() const { return nbytes_; }
    DType       dtype()  const { return dtype_; }
    Device      device() const { return device_; }
private:
    unsigned char* ptr_ = nullptr;       // zero-filled on allocation, from mem::alloc
    std::size_t    nbytes_ = 0;
    DType  dtype_;
    Device device_;
};

/// A contiguous run of floats, for range-for and indexing.
struct FloatSpan {
    float* first = nullptr;
    float* last  = nullptr;
    float* begin() const { return first; }
    float* end()   const { return last; }
    std::size_t size() const { return (std::size_t)(last - first); }
    float& operator[](std::size_t i) const { return first[i]; }
};

// ── Tensor ───────────────────────────────────────────────────────────────────

/// An n-dimensional array: a Storage, plus the offset, shape and strides that
/// say how to walk it.
///
///   element [i0, i1, ...]  lives at  storage[offset + i0*strides[0] + i1*strides[1] + ...]
///
/// A tensor made by a constructor or an op owns fresh, contiguous (row-major)
/// storage. A view -- transpose, permute, reshape -- shares its source's
/// storage and differs only in shape, strides and offset, so taking one is
/// O(1) and writes through it are visible in the source.
///
/// THE GRADIENT CONVENTION: `grad` is a float32 buffer in row-major order
/// over the tensor's own shape, whatever its strides. Every backward_fn in the
/// library writes `grad()[i]` in that order, and that is what lets views and
/// owned tensors share all of that code.
///
/// Fields are private. Ops read elements through data(), which is the raw
/// row-major pointer and is only meaningful for a contiguous float32 tensor;
/// every op therefore runs its inputs through checked_input() first, which
/// materialises views and rejects other dtypes.
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    /// Zero-filled tensor of the given shape.
    explicit Tensor(ShapeArg shape, DType dtype = DType::Float32,
                    Device device = Device::CPU);

    /// Tag for a tensor whose elements are left uninitialised.
    struct Uninitialized {};
    Tensor(ShapeArg shape, Uninitialized, DType dtype = DType::Float32,
           Device device = Device::CPU);

    /// float32 tensor holding `values` in row-major order.
    Tensor(ShapeArg shape, const std::vector<float>& values);

    /// A view onto existing storage. Used by the view ops.
    Tensor(std::shared_ptr<Storage> storage, Shape shape, Shape strides,
           std::int64_t offset);

    // ── shape and layout ─────────────────────────────────────────────────
    const Shape&  shape()   const { return shape_; }
    const Shape&  strides() const { return strides_; }
    int           dim()     const { return (int)shape_.size(); }
    std::int64_t  size(int d) const;            ///< negative d counts from the end
    std::int64_t  numel()   const { return numel_; }
    std::int64_t  num_el()  const { return numel_; }
    std::int64_t  offset()  const { return offset_; }
    bool is_contiguous() const { return contiguous_; }
    bool is_view()       const { return is_view_; }
    static Shape contiguous_strides(const Shape& shape);

    DType  dtype()  const { return storage_->dtype(); }
    Device device() const { return storage_->device(); }
    const std::shared_ptr<Storage>& storage() const { return storage_; }

    // ── elements ─────────────────────────────────────────────────────────
    /// Raw pointer to the first element, as float. Row-major indexing through
    /// it is only correct for a contiguous float32 tensor; see the class note.
    float*       data()       { return static_cast<float*>(ptr_); }
    const float* data() const { return static_cast<const float*>(ptr_); }

    /// Typed pointer to the first element. Raises if T does not match dtype().
    template <typename T> T*       data_ptr();
    template <typename T> const T* data_ptr() const;

    /// The elements of a contiguous float32 tensor, for range-for.
    FloatSpan values();

    /// Every element as a float, in row-major order, whatever the layout or
    /// dtype. A copy.
    std::vector<float> to_vector() const;

    /// Overwrite every element from row-major `v`; the size must match.
    void set_data(const std::vector<float>& v);

    float item() const;                         ///< the single element of a 1-element tensor
    double item_double() const;                 ///< the same, exact for float64 and int64
    float get(std::int64_t row, std::int64_t col) const;   ///< 2-D element, honours strides
    void  set(std::int64_t row, std::int64_t col, float value);
    void  fill(float value);
    void  print() const;

    // ── gradient ─────────────────────────────────────────────────────────
    bool requires_grad() const { return requires_grad_; }
    /// Only floating-point tensors can require a gradient.
    void set_requires_grad(bool r);

    /// The gradient buffer, row-major over this tensor's shape. float32 for
    /// every floating dtype but float64, whose gradient is kept in double --
    /// grad() is null for it; use grad64().
    float*       grad()       { return grad_.empty() ? nullptr : grad_.data(); }
    const float* grad() const { return grad_.empty() ? nullptr : grad_.data(); }
    double*       grad64()       { return grad64_.empty() ? nullptr : grad64_.data(); }
    const double* grad64() const { return grad64_.empty() ? nullptr : grad64_.data(); }
    /// The gradient as doubles, whatever buffer holds it.
    std::vector<double> grad_doubles() const;
    /// The gradient as a tensor: float64 for a float64 tensor, else float32.
    TensorPtr grad_tensor() const;
    /// Add `g` (same number of elements, any float dtype) into the gradient.
    void accumulate_grad(const TensorPtr& g);
    FloatSpan    grad_values();
    std::vector<float> grad_vector() const {
        return std::vector<float>(grad_.data(), grad_.data() + grad_.size());
    }
    void set_grad(const std::vector<float>& g);
    bool has_grad() const {
        return (std::int64_t)(dtype() == DType::Float64 ? grad64_.size() : grad_.size()) == numel_;
    }
    /// Allocate a zeroed gradient buffer if there is not one already.
    void ensure_grad();
    /// Allocate if needed and zero.
    void zero_grad();
    void reset_grad() { zero_grad(); }

    const std::shared_ptr<GradNode>& grad_fn() const { return grad_fn_; }
    void set_grad_fn(std::shared_ptr<GradNode> node) { grad_fn_ = std::move(node); }

    // ── factories ────────────────────────────────────────────────────────
    static TensorPtr zeros(ShapeArg shape, DType dtype = DType::Float32);
    /// Uninitialised: for a kernel output that writes every element. Skips the
    /// zero-fill pass, a third of the memory traffic of a memory-bound op.
    static TensorPtr empty(ShapeArg shape, DType dtype = DType::Float32);
    static TensorPtr ones(ShapeArg shape, DType dtype = DType::Float32);
    static TensorPtr full(ShapeArg shape, float value, DType dtype = DType::Float32);
    /// float32 samples from N(0, 1).
    static TensorPtr randn(ShapeArg shape);
    /// Seed the library-wide RNG so randn() and weight init are reproducible.
    static void manual_seed(unsigned int seed);

    // ── conversion and the graph ─────────────────────────────────────────
    /// A copy in another dtype; the tensor itself when it already is one.
    /// Between floating dtypes the cast is differentiable, as in PyTorch: the
    /// gradient is cast back. To or from an integer or bool dtype it leaves
    /// the graph.
    TensorPtr to(DType dtype);
    /// A copy on `device`. CUDA needs a GPU and its driver (see ml/cuda.hpp);
    /// without them this raises with the reason. Not differentiable yet.
    TensorPtr to(Device device);

    /// Differentiable views. Equivalent to the free functions of the same name.
    TensorPtr transpose(int dim0 = -2, int dim1 = -1);
    TensorPtr reshape(ShapeArg shape);
    TensorPtr contiguous();

    void      backward();
    /// Same values, contiguous, out of the graph.
    TensorPtr detach() const;
    /// float32 with each value rounded through fp16, to simulate half
    /// precision. For real fp16 storage use to(DType::Float16).
    TensorPtr half() const;
    TensorPtr to_float() const;

private:
    void init_layout();

    std::shared_ptr<Storage> storage_;
    void*        ptr_     = nullptr;   // first element
    std::int64_t offset_  = 0;         // in elements
    std::int64_t numel_   = 0;
    Shape        shape_;
    Shape        strides_;
    bool         contiguous_    = true;
    bool         is_view_       = false;
    bool         requires_grad_ = false;
    FloatBuffer               grad_;
    std::vector<double>       grad64_;     // float64 tensors only
    std::shared_ptr<GradNode> grad_fn_;
};

template <typename T> struct dtype_of;
template <> struct dtype_of<float>         { static constexpr DType value = DType::Float32; };
template <> struct dtype_of<double>        { static constexpr DType value = DType::Float64; };
template <> struct dtype_of<std::int64_t>  { static constexpr DType value = DType::Int64; };
template <> struct dtype_of<std::int32_t>  { static constexpr DType value = DType::Int32; };
template <> struct dtype_of<bool>          { static constexpr DType value = DType::Bool; };

template <typename T> T* Tensor::data_ptr() {
    if (dtype() != dtype_of<T>::value)
        throw std::runtime_error(std::string("data_ptr: tensor is ") + dtype_name(dtype()) +
                                 ", not " + dtype_name(dtype_of<T>::value));
    return static_cast<T*>(ptr_);
}
template <typename T> const T* Tensor::data_ptr() const {
    return const_cast<Tensor*>(this)->data_ptr<T>();
}

std::mt19937& prometheus_rng();

/// Global switch for building the graph; see NoGrad.
bool& grad_enabled();

/// Scope that turns graph-building off, like torch.no_grad(). Ops still
/// compute their results but attach no backward node, so inference holds no
/// activations for a backward pass that will never run.
struct NoGrad {
    bool previous;
    NoGrad() : previous(grad_enabled()) { grad_enabled() = false; }
    ~NoGrad() { grad_enabled() = previous; }
};

/// Wire a new result tensor into the computation graph and return its node,
/// for the caller to attach a backward_fn to. Under NoGrad the node is
/// returned but not attached, so call sites need no special case.
inline std::shared_ptr<GradNode> make_node(TensorPtr result, std::vector<TensorPtr> inputs) {
    auto node = std::make_shared<GradNode>();
    node->inputs = std::move(inputs);
    if (!grad_enabled())
        return node;

    // anything joining the graph will be written to by a backward_fn
    for (auto& in : node->inputs)
        if (in && in->requires_grad())
            in->ensure_grad();
    result->ensure_grad();

    result->set_requires_grad(true);
    result->set_grad_fn(node);
    return node;
}
