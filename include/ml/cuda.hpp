#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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

} // namespace cuda
