// Ops on GPU tensors: kernels compiled at runtime by NVRTC. See ml/cuda.hpp.
#include "../ops_internal.hpp"
#include "ml/cuda.hpp"
#include <stdexcept>
#include <string>

using namespace std;

namespace {

constexpr int MAX_DIMS = 8;

// Broadcasting index map, passed to a kernel by value: each output element's
// row-major position decomposes over `shape`, and each input's element is at
// that position dotted with its strides (0 along a broadcast dimension). The
// inputs' own strides are used, so a transposed or sliced view is read in
// place.
struct Index2 {
    long long shape[MAX_DIMS];
    long long sa[MAX_DIMS];
    long long sb[MAX_DIMS];
    int nd;
};

const char* const BINARY_SRC = R"(
#define MAX_DIMS 8
struct Index2 { long long shape[MAX_DIMS]; long long sa[MAX_DIMS]; long long sb[MAX_DIMS]; int nd; };
#define BINARY(NAME, EXPR)                                                                 \
extern "C" __global__ void NAME(const float* a, const float* b, float* c, long long n,     \
                                Index2 ix) {                                               \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;                        \
    if (i >= n) return;                                                                    \
    long long r = i, ia = 0, ib = 0;                                                       \
    for (int d = ix.nd - 1; d >= 0; d--) {                                                 \
        long long q = r % ix.shape[d];                                                     \
        r /= ix.shape[d];                                                                  \
        ia += q * ix.sa[d];                                                                \
        ib += q * ix.sb[d];                                                                \
    }                                                                                      \
    float x = a[ia], y = b[ib];                                                            \
    c[i] = EXPR;                                                                           \
}
BINARY(prom_add, x + y)
BINARY(prom_sub, x - y)
BINARY(prom_mul, x * y)
BINARY(prom_div, x / y)
)";

const cuda::Kernel BINARY_KERNELS[] = {
    {"prom_add", BINARY_SRC}, {"prom_sub", BINARY_SRC},
    {"prom_mul", BINARY_SRC}, {"prom_div", BINARY_SRC},
};

// t's strides laid against `out`: its own, 0 where it is broadcast or missing
void strides_against(const TensorPtr& t, const Shape& out, long long* dst) {
    const int nd = (int)out.size(), off = nd - t->dim();
    for (int d = 0; d < nd; d++) {
        if (d < off) { dst[d] = 0; continue; }
        const int k = d - off;
        dst[d] = t->shape()[k] == 1 ? 0 : t->strides()[k];
    }
}

void require(const TensorPtr& t, const char* op) {
    if (t->device() != Device::CUDA)
        throw runtime_error(string(op) + ": the operands are on different devices; move both "
                            "with .to(cuda) or .to(cpu)");
    if (t->dtype() != DType::Float32)
        throw runtime_error(string(op) + ": on the GPU only float32 is supported so far, got " +
                            dtype_name(t->dtype()));
    if (t->requires_grad())
        throw runtime_error(string(op) + ": gradients on the GPU are not supported yet");
}

} // namespace

namespace cuda_ops {

TensorPtr binary(int op, const char* name, const TensorPtr& a, const TensorPtr& b) {
    require(a, name);
    require(b, name);
    const Shape shape = broadcast_shapes(a->shape(), b->shape());
    if ((int)shape.size() > MAX_DIMS)
        throw runtime_error(string(name) + ": more than 8 dimensions on the GPU");
    auto out = make_shared<Tensor>(shape, Tensor::Uninitialized{}, DType::Float32, Device::CUDA);
    Index2 ix{};
    ix.nd = (int)shape.size();
    for (int d = 0; d < ix.nd; d++) ix.shape[d] = shape[d];
    strides_against(a, shape, ix.sa);
    strides_against(b, shape, ix.sb);
    const float* pa = static_cast<const float*>(a->storage()->data()) + a->offset();
    const float* pb = static_cast<const float*>(b->storage()->data()) + b->offset();
    float* pc = static_cast<float*>(out->storage()->data());
    long long n = out->numel();
    void* args[] = {&pa, &pb, &pc, &n, &ix};
    cuda::launch(BINARY_KERNELS[op], n, args);
    return out;
}

} // namespace cuda_ops
