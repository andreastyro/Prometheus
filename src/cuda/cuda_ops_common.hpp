#pragma once
// Internal to the GPU ops' source files (cuda_ops.cpp, cuda_nn.cpp,
// cuda_vision.cpp, cuda_rnn.cpp): the helpers they share. Not installed.
#include "../ops_internal.hpp"
#include "ml/cuda.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cuda_common {

constexpr int MAX_DIMS = 8;
constexpr DType F32 = DType::Float32;

/// The device-side helpers every kernel source starts with (dtype loads and
/// stores, the strided index map, gid()).
extern const char* const COMMON;

/// A tensor's strides laid against an output shape, dimensions merged; nd = -1
/// for a contiguous operand of the same shape. Passed to kernels by value.
struct Index {
    long long shape[MAX_DIMS];
    long long st[MAX_DIMS];
    int nd;
};
Index index_for(const TensorPtr& t, const Shape& out);

/// COMMON + a kernel source's body
std::string src(const char* body);
cuda::Kernel K(const char* name, const std::string& source);
/// an uninitialised tensor on the device
TensorPtr empty_cuda(const Shape& shape, DType d = F32);
/// a tensor's first element on the device (storage plus offset)
const void* base(const TensorPtr& t);
/// a node whose backward is `graph` (written with ops)
void attach(const TensorPtr& out, std::vector<TensorPtr> inputs, const char* name,
            std::function<std::vector<TensorPtr>(const TensorPtr&, const TensorPtr&)> graph);
TensorPtr contiguous_on_device(const TensorPtr& t);
TensorPtr upload_indices(const std::shared_ptr<const std::vector<int64_t>>& idx);

} // namespace cuda_common

namespace cuda_ops {

using Reference = std::function<TensorPtr(const std::vector<TensorPtr>&)>;
/// operands for a float32 kernel: on the device, as float32 (through cast_op,
/// so a half-precision input's gradient flows back through the conversion)
TensorPtr f32(const TensorPtr& t, const char* op);
/// a node on `out`: `fused` for a plain backward, `reference` (the op form)
/// when the gradient is recorded
void attach_fused(const TensorPtr& out, std::vector<TensorPtr> inputs, const char* name, Reference reference,
                  std::function<std::vector<TensorPtr>(const TensorPtr& G)> fused);

} // namespace cuda_ops
