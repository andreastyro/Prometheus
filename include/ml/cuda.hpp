#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "ml/tensor.hpp"

/// The GPU, loaded at runtime.
///
/// Nothing CUDA is linked and no CUDA SDK is needed to build. When a GPU is
/// first asked for, the library loads the driver that ships with the NVIDIA
/// display driver (nvcuda.dll / libcuda.so.1) and NVRTC, NVIDIA's runtime
/// compiler (pip install nvidia-cuda-nvrtc-cu12). Kernels are CUDA C source
/// strings compiled by NVRTC for the GPU that is actually present the first
/// time each is used, then cached for the life of the process -- the approach
/// CuPy and tinygrad take. One build therefore runs on machines with and
/// without a GPU; on one without, available() is false and asking for a CUDA
/// tensor raises with the reason.
namespace cuda {

/// True when the driver loaded and a GPU was found. Never raises.
bool available();
/// Why not, or the device in use: "GeForce GTX 970, compute 5.2, 4096 MB".
std::string status();

std::string device_name();
/// Tensor cores (compute 7.0+): fp16 matmuls run on them.
bool has_tensor_cores();
/// bfloat16 matmuls in hardware (compute 8.0+). Older GPUs compute bf16
/// matmuls in float32 and round, which is correct but no faster.
bool has_bf16();
/// Compute capability as major*10 + minor (52 for a GTX 970).
int compute_capability();
std::size_t total_memory();

/// Extra directories to search for NVRTC and cuBLAS, ahead of the system's.
/// The Python module adds pip's nvidia/*/bin (Windows) and nvidia/*/lib.
void set_search_dirs(const std::vector<std::string>& dirs);

// ── memory ───────────────────────────────────────────────────────────────────
/// Device memory, from a caching allocator like the CPU one: freed blocks are
/// kept by size and reused, since cuMemAlloc / cuMemFree synchronise the GPU.
void* alloc(std::size_t nbytes);
void  free(void* ptr, std::size_t nbytes) noexcept;
void  memset_zero(void* dst, std::size_t nbytes);
void  copy_to_device(void* dst, const void* src, std::size_t nbytes);
void  copy_to_host(void* dst, const void* src, std::size_t nbytes);
void  copy_device(void* dst, const void* src, std::size_t nbytes);
/// Wait for every queued kernel and copy to finish.
void  synchronize();
/// Bytes of device memory held idle by the allocator.
std::size_t cached_bytes();
void  empty_cache();

/// Where device memory is: held by live tensors (in_use, and its high-water
/// mark since the last reset), idle in the cache, and free on the device.
struct MemoryStats {
    std::size_t in_use = 0, peak = 0, cached = 0, device_free = 0, device_total = 0;
};
MemoryStats memory_stats();
void reset_peak_memory();

/// GPUs the driver can see.
int  device_count();
/// Use GPU `index` (default 0, or $PROMETHEUS_CUDA_DEVICE). Must be called
/// before the first GPU tensor: one device per process.
void set_device(int index);
int  current_device();

// ── kernels ──────────────────────────────────────────────────────────────────
/// A compiled kernel. Compiled from `source` on first use for this GPU's
/// architecture and kept; `name` is the __global__ function to launch, and
/// must be unique per source text.
struct Kernel {
    const char* name;
    const char* source;
};

/// Launch `k` on a 1-D grid covering `n` threads, `block` per block. `args`
/// are pointers to each argument, as cuLaunchKernel takes them.
void launch(const Kernel& k, std::int64_t n, void** args, unsigned block = 256);
/// Launch with an explicit grid and block.
void launch(const Kernel& k, unsigned grid_x, unsigned grid_y, unsigned grid_z,
            unsigned block_x, unsigned block_y, unsigned block_z,
            unsigned shared_bytes, void** args);

// ── cuBLAS ───────────────────────────────────────────────────────────────────
//
// Loaded like NVRTC (pip install nvidia-cublas-cu12). Row-major interface:
// C[m, n] = alpha * op(A) @ op(B) + beta * C, where op(A) is m x k and lda /
// ldb / ldc are row strides, as in gemm_serial. The column-major library is
// driven with the operands swapped, which is the same product transposed.

bool blas_available();
/// Why cuBLAS cannot be used, or the library loaded.
std::string blas_status();

void sgemm(bool trans_a, bool trans_b, int m, int n, int k, float alpha,
           const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc);
/// `batch` products, the i-th using a + i*stride_a, b + i*stride_b, c + i*stride_c.
void sgemm_batched(bool trans_a, bool trans_b, int m, int n, int k, float alpha,
                   const float* a, int lda, long long stride_a,
                   const float* b, int ldb, long long stride_b, float beta,
                   float* c, int ldc, long long stride_c, int batch);

/// The same with float16 / bfloat16 inputs (both of `dtype`) and a float32
/// result, accumulated in float32 -- on tensor cores where the GPU has them.
/// bfloat16 needs has_bf16(). `batch` products as sgemm_batched.
void gemm_half(bool trans_a, bool trans_b, int m, int n, int k, DType dtype,
               const void* a, int lda, long long stride_a,
               const void* b, int ldb, long long stride_b,
               float* c, int ldc, long long stride_c, int batch);

} // namespace cuda
