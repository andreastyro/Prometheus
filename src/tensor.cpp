#include "ml/tensor.hpp"
#include "ml/autograd.hpp"
#include "ml/ops.hpp"
#include "ml/strided.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>

using namespace std;

// ── dtype ────────────────────────────────────────────────────────────────────

size_t dtype_size(DType d) {
    switch (d) {
        case DType::Float32:  return 4;
        case DType::Float64:  return 8;
        case DType::Float16:  return 2;
        case DType::BFloat16: return 2;
        case DType::Int64:    return 8;
        case DType::Int32:    return 4;
        case DType::Bool:     return 1;
    }
    return 4;
}

const char* dtype_name(DType d) {
    switch (d) {
        case DType::Float32:  return "float32";
        case DType::Float64:  return "float64";
        case DType::Float16:  return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int64:    return "int64";
        case DType::Int32:    return "int32";
        case DType::Bool:     return "bool";
    }
    return "?";
}

const char* device_name(Device d) { return d == Device::CPU ? "cpu" : "cuda"; }

bool is_floating(DType d) {
    return d == DType::Float32 || d == DType::Float64 ||
           d == DType::Float16 || d == DType::BFloat16;
}

// fp16: 1 sign, 5 exponent, 10 mantissa bits. Values too small for a normal
// fp16 flush to zero and values too large saturate to infinity.
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0)  return sign;
    if (exp >= 31) return sign | 0x7C00;
    // round to nearest even on the 13 bits dropped
    uint32_t half = mant >> 13, rest = mant & 0x1FFF;
    uint16_t h = sign | (uint16_t)(exp << 10) | (uint16_t)half;
    if (rest > 0x1000 || (rest == 0x1000 && (half & 1))) h++;
    return h;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t x;
    if (exp == 0)       x = sign | (mant << 13);
    else if (exp == 31) x = sign | 0x7F800000 | (mant << 13);
    else                x = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
}

// bf16 is the top half of an fp32: same exponent range, 7 mantissa bits.
static uint16_t f32_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    if ((x & 0x7F800000) == 0x7F800000 && (x & 0x7FFFFF))
        return (uint16_t)((x >> 16) | 0x40);            // keep NaN a NaN
    x += 0x7FFF + ((x >> 16) & 1);                       // round to nearest even
    return (uint16_t)(x >> 16);
}

static float bf16_to_f32(uint16_t h) {
    uint32_t x = (uint32_t)h << 16;
    float f;
    memcpy(&f, &x, 4);
    return f;
}

namespace strided {

double load(const void* base, DType d, int64_t i) {
    switch (d) {
        case DType::Float32:  return static_cast<const float*>(base)[i];
        case DType::Float64:  return static_cast<const double*>(base)[i];
        case DType::Float16:  return f16_to_f32(static_cast<const uint16_t*>(base)[i]);
        case DType::BFloat16: return bf16_to_f32(static_cast<const uint16_t*>(base)[i]);
        case DType::Int64:    return (double)static_cast<const int64_t*>(base)[i];
        case DType::Int32:    return static_cast<const int32_t*>(base)[i];
        case DType::Bool:     return static_cast<const uint8_t*>(base)[i] ? 1.0 : 0.0;
    }
    return 0.0;
}

void store(void* base, DType d, int64_t i, double v) {
    switch (d) {
        case DType::Float32:  static_cast<float*>(base)[i]    = (float)v; break;
        case DType::Float64:  static_cast<double*>(base)[i]   = v; break;
        case DType::Float16:  static_cast<uint16_t*>(base)[i] = f32_to_f16((float)v); break;
        case DType::BFloat16: static_cast<uint16_t*>(base)[i] = f32_to_bf16((float)v); break;
        case DType::Int64:    static_cast<int64_t*>(base)[i]  = (int64_t)v; break;
        case DType::Int32:    static_cast<int32_t*>(base)[i]  = (int32_t)v; break;
        case DType::Bool:     static_cast<uint8_t*>(base)[i]  = v != 0.0; break;
    }
}

} // namespace strided

// ── Storage ──────────────────────────────────────────────────────────────────

Storage::Storage(size_t nbytes, DType dtype, Device device)
    : dtype_(dtype), device_(device) {
    if (device != Device::CPU)
        throw runtime_error(string("device '") + device_name(device) +
                            "' is not available in this build; only cpu is");
    bytes_.assign(nbytes, 0);
}

// ── Tensor: construction and layout ──────────────────────────────────────────

static int64_t product(const Shape& s) {
    int64_t n = 1;
    for (int64_t d : s) {
        if (d < 0) throw runtime_error("tensor shapes cannot have negative sizes");
        n *= d;
    }
    return n;
}

Shape Tensor::contiguous_strides(const Shape& shape) {
    Shape st(shape.size(), 1);
    for (int d = (int)shape.size() - 2; d >= 0; d--)
        st[d] = st[d + 1] * max<int64_t>(shape[d + 1], 1);
    return st;
}

Tensor::Tensor(ShapeArg shape, DType dtype, Device device)
    : shape_(shape.get()) {
    numel_   = product(shape_);
    storage_ = make_shared<Storage>((size_t)numel_ * dtype_size(dtype), dtype, device);
    strides_ = contiguous_strides(shape_);
    init_layout();
}

Tensor::Tensor(ShapeArg shape, const vector<float>& values)
    : Tensor(shape, DType::Float32, Device::CPU) {
    if ((int64_t)values.size() != numel_)
        throw runtime_error("Tensor: " + to_string(values.size()) + " values for a shape of " +
                            to_string(numel_) + " elements");
    if (numel_) memcpy(ptr_, values.data(), (size_t)numel_ * sizeof(float));
}

Tensor::Tensor(shared_ptr<Storage> storage, Shape shape, Shape strides, int64_t offset)
    : storage_(std::move(storage)), offset_(offset),
      shape_(std::move(shape)), strides_(std::move(strides)), is_view_(true) {
    if (strides_.size() != shape_.size())
        throw runtime_error("view: shape and strides have different lengths");
    init_layout();

    // a bad stride or offset here would read outside the buffer silently
    if (numel_ > 0) {
        int64_t lo = offset_, hi = offset_;
        for (size_t d = 0; d < shape_.size(); d++) {
            const int64_t span = (shape_[d] - 1) * strides_[d];
            (span < 0 ? lo : hi) += span;
        }
        const int64_t elems = (int64_t)(storage_->nbytes() / dtype_size(storage_->dtype()));
        if (lo < 0 || hi >= elems)
            throw runtime_error("view: reaches outside its storage");
    }
}

void Tensor::init_layout() {
    numel_ = product(shape_);
    ptr_   = static_cast<unsigned char*>(storage_->data()) +
             offset_ * (int64_t)dtype_size(storage_->dtype());
    // sizes of 1 can carry any stride without changing where anything is
    contiguous_ = true;
    int64_t expect = 1;
    for (int d = (int)shape_.size() - 1; d >= 0; d--) {
        if (shape_[d] != 1 && strides_[d] != expect) { contiguous_ = false; break; }
        expect *= shape_[d];
    }
}

int64_t Tensor::size(int d) const {
    const int nd = dim();
    if (d < 0) d += nd;
    if (d < 0 || d >= nd)
        throw runtime_error("size: dimension " + to_string(d) + " out of range for " +
                            to_string(nd) + "-D tensor");
    return shape_[d];
}

// ── elements ─────────────────────────────────────────────────────────────────

FloatSpan Tensor::values() {
    if (dtype() != DType::Float32 || !contiguous_)
        throw runtime_error("values(): needs a contiguous float32 tensor; use to_vector()");
    return {data(), data() + numel_};
}

vector<float> Tensor::to_vector() const {
    vector<float> out((size_t)numel_);
    if (contiguous_ && dtype() == DType::Float32) {
        if (numel_) memcpy(out.data(), ptr_, (size_t)numel_ * sizeof(float));
        return out;
    }
    const void* base = storage_->data();
    const DType d = dtype();
    strided::for_each(shape_, strides_, offset_, [&](int64_t i, int64_t off) {
        out[i] = (float)strided::load(base, d, off);
    });
    return out;
}

void Tensor::set_data(const vector<float>& v) {
    if ((int64_t)v.size() != numel_)
        throw runtime_error("set_data: " + to_string(v.size()) + " values for " +
                            to_string(numel_) + " elements");
    if (contiguous_ && dtype() == DType::Float32) {
        if (numel_) memcpy(ptr_, v.data(), (size_t)numel_ * sizeof(float));
        return;
    }
    void* base = storage_->data();
    const DType d = dtype();
    strided::for_each(shape_, strides_, offset_, [&](int64_t i, int64_t off) {
        strided::store(base, d, off, v[i]);
    });
}

float Tensor::item() const {
    if (numel_ != 1)
        throw runtime_error("item(): tensor has " + to_string(numel_) + " elements, not 1");
    return (float)strided::load(storage_->data(), dtype(), offset_);
}

float Tensor::get(int64_t row, int64_t col) const {
    if (dim() != 2) throw runtime_error("get: needs a 2-D tensor");
    return (float)strided::load(storage_->data(), dtype(),
                                offset_ + row * strides_[0] + col * strides_[1]);
}

void Tensor::set(int64_t row, int64_t col, float value) {
    if (dim() != 2) throw runtime_error("set: needs a 2-D tensor");
    strided::store(storage_->data(), dtype(),
                   offset_ + row * strides_[0] + col * strides_[1], value);
}

void Tensor::fill(float value) {
    if (contiguous_ && dtype() == DType::Float32) {
        std::fill(data(), data() + numel_, value);
        return;
    }
    void* base = storage_->data();
    const DType d = dtype();
    strided::for_each(shape_, strides_, offset_, [&](int64_t, int64_t off) {
        strided::store(base, d, off, value);
    });
}

void Tensor::print() const {
    const vector<float> v = to_vector();
    const int64_t cols = dim() ? shape_.back() : 1;
    printf("tensor(shape=[");
    for (int d = 0; d < dim(); d++) printf(d ? ", %lld" : "%lld", (long long)shape_[d]);
    printf("], dtype=%s)\n", dtype_name(dtype()));
    for (int64_t i = 0; i < (int64_t)v.size(); i++) {
        printf("%10.4f ", v[i]);
        if (cols && (i + 1) % cols == 0) printf("\n");
    }
}

// ── gradient ─────────────────────────────────────────────────────────────────

void Tensor::set_requires_grad(bool r) {
    if (r && !is_floating(dtype()))
        throw runtime_error(string("only floating-point tensors can require a gradient; "
                                   "this one is ") + dtype_name(dtype()));
    requires_grad_ = r;
}

FloatSpan Tensor::grad_values() {
    return grad_.empty() ? FloatSpan{} : FloatSpan{grad_.data(), grad_.data() + grad_.size()};
}

void Tensor::set_grad(const vector<float>& g) {
    if ((int64_t)g.size() != numel_)
        throw runtime_error("set_grad: " + to_string(g.size()) + " values for " +
                            to_string(numel_) + " elements");
    grad_ = g;
}

void Tensor::ensure_grad() {
    if ((int64_t)grad_.size() != numel_)
        grad_.assign((size_t)numel_, 0.0f);
}

void Tensor::zero_grad() {
    grad_.assign((size_t)numel_, 0.0f);
}

// ── factories ────────────────────────────────────────────────────────────────

static mt19937& global_rng() {
    static mt19937 gen(random_device{}());
    return gen;
}

mt19937& prometheus_rng() { return global_rng(); }

bool& grad_enabled() {
    static bool enabled = true;
    return enabled;
}

void Tensor::manual_seed(unsigned int seed) { global_rng().seed(seed); }

TensorPtr Tensor::zeros(ShapeArg shape, DType dtype) {
    return make_shared<Tensor>(shape, dtype);
}

TensorPtr Tensor::ones(ShapeArg shape, DType dtype) {
    return full(shape, 1.0f, dtype);
}

TensorPtr Tensor::full(ShapeArg shape, float value, DType dtype) {
    auto t = make_shared<Tensor>(shape, dtype);
    t->fill(value);
    return t;
}

TensorPtr Tensor::randn(ShapeArg shape) {
    auto t = make_shared<Tensor>(shape);
    normal_distribution<float> dist(0.0f, 1.0f);
    for (float& x : t->values())
        x = dist(global_rng());
    return t;
}

// ── conversion and the graph ─────────────────────────────────────────────────

TensorPtr Tensor::to(DType target) {
    if (target == dtype())
        return shared_from_this();
    auto out = make_shared<Tensor>(shape_, target);
    const void* src = storage_->data();
    void* dst = out->storage()->data();
    const DType from = dtype();
    strided::for_each(shape_, strides_, offset_, [&](int64_t i, int64_t off) {
        strided::store(dst, target, i, strided::load(src, from, off));
    });
    return out;
}

TensorPtr Tensor::to(Device target) {
    if (target == device())
        return shared_from_this();
    throw runtime_error(string("device '") + device_name(target) +
                        "' is not available in this build; only cpu is");
}

TensorPtr Tensor::transpose(int dim0, int dim1) { return ::transpose(shared_from_this(), dim0, dim1); }
TensorPtr Tensor::reshape(ShapeArg shape)       { return ::reshape(shared_from_this(), shape); }
TensorPtr Tensor::contiguous()                  { return ::contiguous(shared_from_this()); }

static void dfs(TensorPtr node, vector<TensorPtr>& order, unordered_set<TensorPtr>& visited) {
    // iterative: a deep graph (a long unrolled RNN) would overflow the stack
    vector<pair<TensorPtr, bool>> stack;      // (node, children_expanded)
    stack.push_back({node, false});
    while (!stack.empty()) {
        auto [cur, expanded] = stack.back();
        stack.pop_back();
        if (expanded) { order.push_back(cur); continue; }
        if (!cur || visited.count(cur)) continue;
        visited.insert(cur);
        stack.push_back({cur, true});         // emit after its inputs
        if (cur->grad_fn())
            for (auto& input : cur->grad_fn()->inputs)
                if (input && !visited.count(input))
                    stack.push_back({input, false});
    }
}

void Tensor::backward() {
    if (!grad_fn_)
        throw runtime_error(
            "backward() called on a tensor with no computation graph. It is "
            "either a leaf, or it was produced while gradients were disabled "
            "(NoGrad / no_grad()).");

    grad_.assign((size_t)numel_, 1.0f);

    vector<TensorPtr> order;
    unordered_set<TensorPtr> visited;
    dfs(shared_from_this(), order, visited);
    reverse(order.begin(), order.end());

    for (TensorPtr& t : order)
        if (t->grad_fn())
            t->grad_fn()->backward_fn(t);
}

TensorPtr Tensor::detach() const {
    auto t = make_shared<Tensor>(shape_, dtype());
    const void* src = storage_->data();
    void* dst = t->storage()->data();
    const DType d = dtype();
    const size_t es = dtype_size(d);
    if (contiguous_) {
        if (numel_) memcpy(dst, ptr_, (size_t)numel_ * es);
    } else {
        strided::for_each(shape_, strides_, offset_, [&](int64_t i, int64_t off) {
            memcpy(static_cast<unsigned char*>(dst) + i * es,
                   static_cast<const unsigned char*>(src) + off * es, es);
        });
    }
    return t;
}

TensorPtr Tensor::half() const {
    vector<float> v = to_vector();
    for (float& x : v) x = f16_to_f32(f32_to_f16(x));
    return make_shared<Tensor>(shape_, v);
}

TensorPtr Tensor::to_float() const {
    return make_shared<Tensor>(shape_, to_vector());
}
