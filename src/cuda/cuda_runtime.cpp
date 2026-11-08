// The CUDA driver and NVRTC, loaded at runtime. See ml/cuda.hpp.
//
// Only the handful of driver API entry points used are declared, by hand, with
// the ABI types they have in cuda.h; nothing here includes a CUDA header.
#include "ml/cuda.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

using namespace std;

namespace cuda {
namespace {

// ── driver API types and entry points ────────────────────────────────────────
using CUresult    = int;
using CUdevice    = int;
using CUdeviceptr = unsigned long long;
using CUcontext   = void*;
using CUmodule    = void*;
using CUfunction  = void*;
using CUstream    = void*;
constexpr int ATTR_CC_MAJOR = 75, ATTR_CC_MINOR = 76;

struct Driver {
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGetCount)(int*);
    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuDeviceGetName)(char*, int, CUdevice);
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice);
    CUresult (*cuDeviceTotalMem)(size_t*, CUdevice);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuCtxSynchronize)();
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemsetD8)(CUdeviceptr, unsigned char, size_t);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
    CUresult (*cuMemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
    CUresult (*cuModuleLoadData)(CUmodule*, const void*);
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                               unsigned, unsigned, CUstream, void**, void**);
    CUresult (*cuGetErrorString)(CUresult, const char**);
    CUresult (*cuMemGetInfo)(size_t*, size_t*);
    CUresult (*cuStreamCreate)(CUstream*, unsigned);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
    CUresult (*cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
    CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    CUresult (*cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream);
};

// NVRTC: the runtime compiler
using nvrtcProgram = void*;
struct Nvrtc {
    int (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
    int (*nvrtcCompileProgram)(nvrtcProgram, int, const char* const*);
    int (*nvrtcGetPTXSize)(nvrtcProgram, size_t*);
    int (*nvrtcGetPTX)(nvrtcProgram, char*);
    int (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t*);
    int (*nvrtcGetProgramLog)(nvrtcProgram, char*);
    int (*nvrtcDestroyProgram)(nvrtcProgram*);
    const char* (*nvrtcGetErrorString)(int);
    int (*nvrtcGetNumSupportedArchs)(int*);          // optional (CUDA 11.2+)
    int (*nvrtcGetSupportedArchs)(int*);             // optional
};

struct State {
    bool        tried = false, ok = false;
    string      why = "not initialised";
    Driver      d{};
    CUdevice    dev = 0;
    CUcontext   ctx = nullptr;
    // All work goes on one non-blocking stream, in order. Kernels and cuBLAS
    // calls were already asynchronous; the copies and memsets on the legacy
    // default stream were not, and each made the CPU wait for the GPU to
    // drain. Now the CPU only waits when it reads a value back.
    CUstream    stream = nullptr;
    string      name;
    int         cc = 0;
    size_t      mem = 0;
    bool        nvrtc_tried = false, nvrtc_ok = false;
    string      nvrtc_why;
    Nvrtc       r{};
    int         arch = 0;       // the compute_XY NVRTC compiles for
};

State                         g;
mutex                         g_mu;
vector<string>                g_dirs;
// keyed by the name and source pointers: both are string literals or static
// strings, so the pair identifies the kernel without hashing its text
struct KernelKey {
    const char* name;
    const char* source;
    bool operator==(const KernelKey& o) const { return name == o.name && source == o.source; }
};
struct KernelKeyHash {
    size_t operator()(const KernelKey& k) const {
        return std::hash<const void*>()(k.name) * 31u ^ std::hash<const void*>()(k.source);
    }
};
unordered_map<KernelKey, CUfunction, KernelKeyHash> g_kernels;
mutex                         g_kernel_mu;
map<size_t, vector<CUdeviceptr>> g_pool;   // idle blocks by size
size_t                        g_pool_bytes = 0;
size_t                        g_in_use = 0, g_peak = 0;   // bytes handed to tensors
int                           g_device_index = -1;        // -1: not chosen yet
mutex                         g_pool_mu;

void* open_library(const string& file) {
#ifdef _WIN32
    if (file.find_first_of("\\/") != string::npos) {
        // LOAD_WITH_ALTERED_SEARCH_PATH needs an absolute path with backslashes
        string p = file;
        replace(p.begin(), p.end(), '/', '\\');
        return (void*)LoadLibraryExA(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    return (void*)LoadLibraryA(file.c_str());
#else
    return dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* symbol(void* lib, const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

void* open_first(const vector<string>& names) {
    for (const auto& dir : g_dirs)
        for (const auto& n : names)
            if (void* h = open_library(dir + "/" + n)) return h;
    for (const auto& n : names)
        if (void* h = open_library(n)) return h;
    return nullptr;
}

template <class F> bool bind(void* lib, F& fn, const char* name) {
    fn = reinterpret_cast<F>(symbol(lib, name));
    return fn != nullptr;
}

string err(CUresult r) {
    const char* s = nullptr;
    if (g.d.cuGetErrorString) g.d.cuGetErrorString(r, &s);
    return s ? s : ("CUDA error " + to_string(r));
}

void check(CUresult r, const char* what) {
    if (r != 0) throw runtime_error(string("cuda: ") + what + " failed: " + err(r));
}

void init_locked() {
    if (g.tried) return;
    g.tried = true;
#ifdef _WIN32
    void* lib = open_first({"nvcuda.dll"});
#else
    void* lib = open_first({"libcuda.so.1", "libcuda.so"});
#endif
    if (!lib) { g.why = "no NVIDIA driver found (nvcuda.dll / libcuda.so.1)"; return; }
    Driver& d = g.d;
    const bool all =
        bind(lib, d.cuInit, "cuInit") && bind(lib, d.cuDeviceGetCount, "cuDeviceGetCount") &&
        bind(lib, d.cuDeviceGet, "cuDeviceGet") && bind(lib, d.cuDeviceGetName, "cuDeviceGetName") &&
        bind(lib, d.cuDeviceGetAttribute, "cuDeviceGetAttribute") &&
        bind(lib, d.cuDeviceTotalMem, "cuDeviceTotalMem_v2") &&
        bind(lib, d.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain") &&
        bind(lib, d.cuCtxSetCurrent, "cuCtxSetCurrent") && bind(lib, d.cuCtxSynchronize, "cuCtxSynchronize") &&
        bind(lib, d.cuMemAlloc, "cuMemAlloc_v2") && bind(lib, d.cuMemFree, "cuMemFree_v2") &&
        bind(lib, d.cuMemsetD8, "cuMemsetD8_v2") && bind(lib, d.cuMemcpyHtoD, "cuMemcpyHtoD_v2") &&
        bind(lib, d.cuMemcpyDtoH, "cuMemcpyDtoH_v2") && bind(lib, d.cuMemcpyDtoD, "cuMemcpyDtoD_v2") &&
        bind(lib, d.cuModuleLoadData, "cuModuleLoadData") &&
        bind(lib, d.cuModuleGetFunction, "cuModuleGetFunction") &&
        bind(lib, d.cuLaunchKernel, "cuLaunchKernel") && bind(lib, d.cuGetErrorString, "cuGetErrorString") &&
        bind(lib, d.cuMemGetInfo, "cuMemGetInfo_v2") &&
        bind(lib, d.cuStreamCreate, "cuStreamCreate") && bind(lib, d.cuStreamSynchronize, "cuStreamSynchronize") &&
        bind(lib, d.cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2") && bind(lib, d.cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2") &&
        bind(lib, d.cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2") && bind(lib, d.cuMemsetD8Async, "cuMemsetD8Async");
    if (!all) { g.why = "the NVIDIA driver is too old (a driver API entry point is missing)"; return; }
    CUresult r = d.cuInit(0);
    if (r != 0) { g.why = "cuInit failed: " + err(r); return; }
    int count = 0;
    d.cuDeviceGetCount(&count);
    if (count < 1) { g.why = "the NVIDIA driver found no GPU"; return; }
    if (g_device_index < 0) {
        const char* env = getenv("PROMETHEUS_CUDA_DEVICE");
        g_device_index = env ? atoi(env) : 0;
    }
    if (g_device_index >= count) {
        g.why = "GPU " + to_string(g_device_index) + " asked for, but only " + to_string(count) + " present";
        return;
    }
    if ((r = d.cuDeviceGet(&g.dev, g_device_index)) != 0) { g.why = "cuDeviceGet failed: " + err(r); return; }
    char name[256] = {0};
    d.cuDeviceGetName(name, sizeof name, g.dev);
    g.name = name;
    int major = 0, minor = 0;
    d.cuDeviceGetAttribute(&major, ATTR_CC_MAJOR, g.dev);
    d.cuDeviceGetAttribute(&minor, ATTR_CC_MINOR, g.dev);
    g.cc = major * 10 + minor;
    d.cuDeviceTotalMem(&g.mem, g.dev);
    if ((r = d.cuDevicePrimaryCtxRetain(&g.ctx, g.dev)) != 0) { g.why = "no CUDA context: " + err(r); return; }
    if ((r = d.cuCtxSetCurrent(g.ctx)) != 0 || (r = d.cuStreamCreate(&g.stream, 1 /* non-blocking */)) != 0) {
        g.why = "no CUDA stream: " + err(r);
        return;
    }
    g.ok = true;
    g.why.clear();
}

State& state() {
    lock_guard<mutex> lock(g_mu);
    init_locked();
    return g;
}

// every thread that touches the GPU makes the context current once
void bind_context() {
    State& s = state();
    if (!s.ok) throw runtime_error("cuda: not available: " + s.why);
    thread_local bool bound = false;
    if (!bound) { check(s.d.cuCtxSetCurrent(s.ctx), "cuCtxSetCurrent"); bound = true; }
}

void init_nvrtc_locked() {
    if (g.nvrtc_tried) return;
    g.nvrtc_tried = true;
#ifdef _WIN32
    void* lib = open_first({"nvrtc64_120_0.dll", "nvrtc64_130_0.dll", "nvrtc64_112_0.dll"});
#else
    void* lib = open_first({"libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so.11.2", "libnvrtc.so"});
#endif
    if (!lib) {
        g.nvrtc_why = "NVRTC not found; pip install nvidia-cuda-nvrtc-cu12";
        return;
    }
    Nvrtc& r = g.r;
    const bool all =
        bind(lib, r.nvrtcCreateProgram, "nvrtcCreateProgram") &&
        bind(lib, r.nvrtcCompileProgram, "nvrtcCompileProgram") &&
        bind(lib, r.nvrtcGetPTXSize, "nvrtcGetPTXSize") && bind(lib, r.nvrtcGetPTX, "nvrtcGetPTX") &&
        bind(lib, r.nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize") &&
        bind(lib, r.nvrtcGetProgramLog, "nvrtcGetProgramLog") &&
        bind(lib, r.nvrtcDestroyProgram, "nvrtcDestroyProgram") &&
        bind(lib, r.nvrtcGetErrorString, "nvrtcGetErrorString");
    if (!all) { g.nvrtc_why = "NVRTC is missing an entry point"; return; }
    bind(lib, r.nvrtcGetNumSupportedArchs, "nvrtcGetNumSupportedArchs");
    bind(lib, r.nvrtcGetSupportedArchs, "nvrtcGetSupportedArchs");

    // the newest architecture NVRTC can target that this GPU can run
    g.arch = g.cc;
    if (r.nvrtcGetNumSupportedArchs && r.nvrtcGetSupportedArchs) {
        int n = 0;
        r.nvrtcGetNumSupportedArchs(&n);
        vector<int> archs((size_t)max(n, 0));
        if (n > 0) r.nvrtcGetSupportedArchs(archs.data());
        int best = 0;
        for (int a : archs) if (a <= g.cc) best = max(best, a);
        if (!best) {
            g.nvrtc_why = "this NVRTC cannot compile for compute " + to_string(g.cc) +
                          " (CUDA 13 dropped Maxwell and Pascal; use nvidia-cuda-nvrtc-cu12)";
            return;
        }
        g.arch = best;
    }
    g.nvrtc_ok = true;
}

CUfunction compile(const Kernel& k) {
    {
        lock_guard<mutex> lock(g_mu);
        init_nvrtc_locked();
        if (!g.nvrtc_ok) throw runtime_error("cuda: cannot compile kernels: " + g.nvrtc_why);
    }
    const Nvrtc& r = g.r;
    nvrtcProgram prog = nullptr;
    int rc = r.nvrtcCreateProgram(&prog, k.source, k.name, 0, nullptr, nullptr);
    if (rc != 0) throw runtime_error(string("cuda: nvrtcCreateProgram: ") + r.nvrtcGetErrorString(rc));
    const string arch = "--gpu-architecture=compute_" + to_string(g.arch);
    const char* opts[] = {arch.c_str(), "--std=c++14", "-default-device"};
    rc = r.nvrtcCompileProgram(prog, 3, opts);
    if (rc != 0) {
        size_t n = 0;
        r.nvrtcGetProgramLogSize(prog, &n);
        string log(n, '\0');
        if (n) r.nvrtcGetProgramLog(prog, &log[0]);
        r.nvrtcDestroyProgram(&prog);
        throw runtime_error(string("cuda: compiling kernel '") + k.name + "' failed:\n" + log);
    }
    size_t n = 0;
    r.nvrtcGetPTXSize(prog, &n);
    string ptx(n, '\0');
    r.nvrtcGetPTX(prog, &ptx[0]);
    r.nvrtcDestroyProgram(&prog);

    CUmodule mod = nullptr;
    check(g.d.cuModuleLoadData(&mod, ptx.c_str()), "cuModuleLoadData");
    CUfunction fn = nullptr;
    check(g.d.cuModuleGetFunction(&fn, mod, k.name), "cuModuleGetFunction");
    return fn;
}

CUfunction function(const Kernel& k) {
    bind_context();
    const KernelKey key{k.name, k.source};
    {
        lock_guard<mutex> lock(g_kernel_mu);
        auto it = g_kernels.find(key);
        if (it != g_kernels.end()) return it->second;
    }
    CUfunction fn = compile(k);
    lock_guard<mutex> lock(g_kernel_mu);
    g_kernels[key] = fn;
    return fn;
}

// requests round up to 512 bytes so near-equal sizes share blocks
size_t bucket(size_t n) { return (n + 511) & ~size_t(511); }

} // namespace

// ── public ───────────────────────────────────────────────────────────────────

bool available() {
    try { return state().ok; } catch (...) { return false; }
}

string status() {
    State& s = state();
    if (!s.ok) return "unavailable: " + s.why;
    return s.name + ", compute " + to_string(s.cc / 10) + "." + to_string(s.cc % 10) + ", " +
           to_string(s.mem >> 20) + " MB";
}

string device_name()     { return state().name; }
int compute_capability() { return state().cc; }
size_t total_memory()    { return state().mem; }

void set_search_dirs(const vector<string>& dirs) {
    lock_guard<mutex> lock(g_mu);
    g_dirs.clear();
    for (string d : dirs) {      // `set X=dir && ...` in cmd leaves a trailing space
        while (!d.empty() && (d.back() == ' ' || d.back() == '\t')) d.pop_back();
        while (!d.empty() && (d.front() == ' ' || d.front() == '\t')) d.erase(d.begin());
        if (!d.empty()) g_dirs.push_back(d);
    }
#ifdef _WIN32
    // NVRTC loads its builtins library by name, from beside itself or PATH
    vector<char> buf(32768);
    DWORD n = GetEnvironmentVariableA("PATH", buf.data(), (DWORD)buf.size());
    string path(buf.data(), n < buf.size() ? n : 0);
    for (const auto& d : g_dirs)
        if (path.find(d) == string::npos) path = d + ";" + path;
    SetEnvironmentVariableA("PATH", path.c_str());
#endif
}

void* alloc(size_t nbytes) {
    if (nbytes == 0) return nullptr;
    bind_context();
    const size_t b = bucket(nbytes);
    {
        lock_guard<mutex> lock(g_pool_mu);
        auto it = g_pool.find(b);
        if (it != g_pool.end() && !it->second.empty()) {
            CUdeviceptr p = it->second.back();
            it->second.pop_back();
            g_pool_bytes -= b;
            g_in_use += b;
            g_peak = max(g_peak, g_in_use);
            return (void*)p;
        }
    }
    CUdeviceptr p = 0;
    CUresult r = g.d.cuMemAlloc(&p, b);
    size_t released = 0;
    if (r != 0) {            // out of memory: give the cache back and retry once
        released = cached_bytes();
        empty_cache();
        r = g.d.cuMemAlloc(&p, b);
    }
    if (r != 0) {
        size_t free_b = 0, total_b = 0;
        g.d.cuMemGetInfo(&free_b, &total_b);
        auto mb = [](size_t n) { return to_string((n + (1 << 19)) >> 20) + " MB"; };
        size_t used;
        {
            lock_guard<mutex> lock(g_pool_mu);
            used = g_in_use;
        }
        throw runtime_error("cuda: out of GPU memory allocating " + mb(b) + " on " + g.name + ": " +
                            mb(used) + " held by tensors, " + mb(free_b) + " free of " + mb(total_b) +
                            (released ? " (after releasing " + mb(released) + " of cached blocks)" : "") +
                            ". Reduce the batch size or sequence length, or free tensors you no longer need.");
    }
    lock_guard<mutex> lock(g_pool_mu);
    g_in_use += b;
    g_peak = max(g_peak, g_in_use);
    return (void*)p;
}

void free(void* ptr, size_t nbytes) noexcept {
    if (!ptr) return;
    lock_guard<mutex> lock(g_pool_mu);
    const size_t b = bucket(nbytes);
    g_pool[b].push_back((CUdeviceptr)ptr);
    g_pool_bytes += b;
    g_in_use -= min(g_in_use, b);
}

MemoryStats memory_stats() {
    MemoryStats s;
    if (!available()) return s;
    {
        lock_guard<mutex> lock(g_pool_mu);
        s.in_use = g_in_use;
        s.peak = g_peak;
        s.cached = g_pool_bytes;
    }
    bind_context();
    g.d.cuMemGetInfo(&s.device_free, &s.device_total);
    return s;
}

void reset_peak_memory() {
    lock_guard<mutex> lock(g_pool_mu);
    g_peak = g_in_use;
}

int device_count() {
    if (!available()) return 0;
    int n = 0;
    g.d.cuDeviceGetCount(&n);
    return n;
}

void set_device(int index) {
    lock_guard<mutex> lock(g_mu);
    if (index < 0) throw runtime_error("cuda: device index must be 0 or more");
    if (g.tried && g.ok && index != g_device_index)
        throw runtime_error("cuda: set_device(" + to_string(index) + ") after GPU " + to_string(g_device_index) +
                            " is in use; choose the device before the first GPU tensor");
    g_device_index = index;
}

int current_device() {
    return available() ? g_device_index : -1;
}

size_t cached_bytes() {
    lock_guard<mutex> lock(g_pool_mu);
    return g_pool_bytes;
}

void empty_cache() {
    if (!available()) return;
    bind_context();
    lock_guard<mutex> lock(g_pool_mu);
    g.d.cuCtxSynchronize();
    for (auto& [size, blocks] : g_pool)
        for (CUdeviceptr p : blocks) g.d.cuMemFree(p);
    g_pool.clear();
    g_pool_bytes = 0;
}

void memset_zero(void* dst, size_t nbytes) {
    if (!nbytes) return;
    bind_context();
    check(g.d.cuMemsetD8Async((CUdeviceptr)dst, 0, nbytes, g.stream), "cuMemsetD8");
}

void copy_to_device(void* dst, const void* src, size_t nbytes) {
    if (!nbytes) return;
    bind_context();
    // from pageable memory this returns once the bytes are staged, so the
    // caller may free its buffer straight away
    check(g.d.cuMemcpyHtoDAsync((CUdeviceptr)dst, src, nbytes, g.stream), "cuMemcpyHtoD");
}

void copy_to_host(void* dst, const void* src, size_t nbytes) {
    if (!nbytes) return;
    bind_context();
    // the CPU needs the bytes: wait for the stream, and only it
    check(g.d.cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, nbytes, g.stream), "cuMemcpyDtoH");
    check(g.d.cuStreamSynchronize(g.stream), "cuStreamSynchronize");
}

void copy_device(void* dst, const void* src, size_t nbytes) {
    if (!nbytes) return;
    bind_context();
    check(g.d.cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, nbytes, g.stream), "cuMemcpyDtoD");
}

void synchronize() {
    bind_context();
    check(g.d.cuStreamSynchronize(g.stream), "cuStreamSynchronize");
}

// ── cuBLAS ───────────────────────────────────────────────────────────────────

namespace {
using cublasHandle_t = void*;
struct Cublas {
    int (*create)(cublasHandle_t*);
    int (*sgemm)(cublasHandle_t, int, int, int, int, int, const float*, const float*, int,
                 const float*, int, const float*, float*, int);
    int (*sgemm_batched)(cublasHandle_t, int, int, int, int, int, const float*, const float*, int,
                         long long, const float*, int, long long, const float*, float*, int,
                         long long, int);
    // cublasGemmStridedBatchedEx: typed operands, a compute type and an algorithm
    int (*gemm_ex)(cublasHandle_t, int, int, int, int, int, const void*, const void*, int, int,
                   long long, const void*, int, int, long long, const void*, void*, int, int,
                   long long, int, int, int);
    int (*set_stream)(cublasHandle_t, void*);
};
Cublas         g_blas{};
cublasHandle_t g_handle = nullptr;
bool           g_blas_tried = false, g_blas_ok = false;
string         g_blas_why;

void init_blas_locked() {
    if (g_blas_tried) return;
    g_blas_tried = true;
    init_locked();
    if (!g.ok) { g_blas_why = g.why; return; }
#ifdef _WIN32
    void* lib = open_first({"cublas64_12.dll", "cublas64_13.dll", "cublas64_11.dll"});
#else
    void* lib = open_first({"libcublas.so.12", "libcublas.so.13", "libcublas.so.11", "libcublas.so"});
#endif
    if (!lib) { g_blas_why = "cuBLAS not found; pip install nvidia-cublas-cu12"; return; }
    if (!bind(lib, g_blas.create, "cublasCreate_v2") || !bind(lib, g_blas.sgemm, "cublasSgemm_v2") ||
        !bind(lib, g_blas.sgemm_batched, "cublasSgemmStridedBatched") ||
        !bind(lib, g_blas.gemm_ex, "cublasGemmStridedBatchedEx") ||
        !bind(lib, g_blas.set_stream, "cublasSetStream_v2")) {
        g_blas_why = "cuBLAS is missing an entry point";
        return;
    }
    // the handle belongs to the current context
    if (g.d.cuCtxSetCurrent(g.ctx) != 0 || g_blas.create(&g_handle) != 0 ||
        g_blas.set_stream(g_handle, g.stream) != 0) {
        g_blas_why = "cublasCreate failed";
        return;
    }
    g_blas_ok = true;
}

cublasHandle_t blas_handle() {
    bind_context();
    lock_guard<mutex> lock(g_mu);
    init_blas_locked();
    if (!g_blas_ok) throw runtime_error("cuda: matmul needs cuBLAS: " + g_blas_why);
    return g_handle;
}

void check_blas(int status, const char* what) {
    if (status != 0) throw runtime_error(string("cuBLAS ") + what + " failed with status " + to_string(status));
}
} // namespace

bool blas_available() {
    if (!available()) return false;
    lock_guard<mutex> lock(g_mu);
    init_blas_locked();
    return g_blas_ok;
}

string blas_status() {
    if (!available()) return "unavailable: " + state().why;
    lock_guard<mutex> lock(g_mu);
    init_blas_locked();
    return g_blas_ok ? "cuBLAS loaded" : "unavailable: " + g_blas_why;
}

// Row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T: the same
// call with the operands, their flags and m / n swapped.
void sgemm(bool ta, bool tb, int m, int n, int k, float alpha, const float* a, int lda,
           const float* b, int ldb, float beta, float* c, int ldc) {
    if (m == 0 || n == 0) return;
    auto h = blas_handle();
    check_blas(g_blas.sgemm(h, tb ? 1 : 0, ta ? 1 : 0, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc), "sgemm");
}

void sgemm_batched(bool ta, bool tb, int m, int n, int k, float alpha, const float* a, int lda,
                   long long sa, const float* b, int ldb, long long sb, float beta, float* c, int ldc,
                   long long sc, int batch) {
    if (m == 0 || n == 0 || batch == 0) return;
    auto h = blas_handle();
    check_blas(g_blas.sgemm_batched(h, tb ? 1 : 0, ta ? 1 : 0, n, m, k, &alpha, b, ldb, sb, a, lda, sa,
                                    &beta, c, ldc, sc, batch), "sgemm (batched)");
}

bool has_tensor_cores() { return available() && state().cc >= 70; }
bool has_bf16()         { return available() && state().cc >= 80; }

void gemm_half(bool ta, bool tb, int m, int n, int k, DType dtype, const void* a, int lda, long long sa,
               const void* b, int ldb, long long sb, float* c, int ldc, long long sc, int batch) {
    if (m == 0 || n == 0 || batch == 0) return;
    // cudaDataType: CUDA_R_32F 0, CUDA_R_16F 2, CUDA_R_16BF 14; CUBLAS_COMPUTE_32F 68
    const int type = dtype == DType::Float16 ? 2 : dtype == DType::BFloat16 ? 14 : -1;
    if (type < 0) throw runtime_error("gemm_half: inputs must be float16 or bfloat16");
    if (type == 14 && !has_bf16())
        throw runtime_error("gemm_half: bfloat16 matmuls need compute 8.0 or newer");
    auto h = blas_handle();
    const float alpha = 1.0f, beta = 0.0f;
    check_blas(g_blas.gemm_ex(h, tb ? 1 : 0, ta ? 1 : 0, n, m, k, &alpha, b, type, ldb, sb, a, type, lda, sa,
                              &beta, c, 0, ldc, sc, batch, 68, -1), "GemmEx");
}

void launch(const Kernel& k, int64_t n, void** args, unsigned block) {
    if (n <= 0) return;
    const int64_t grid = (n + block - 1) / block;
    launch(k, (unsigned)min<int64_t>(grid, 0x7fffffff), 1, 1, block, 1, 1, 0, args);
}

void launch(const Kernel& k, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
            unsigned bz, unsigned shared, void** args) {
    CUfunction fn = function(k);
    check(g.d.cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, shared, g.stream, args, nullptr),
          (string("launching ") + k.name).c_str());
}

} // namespace cuda
