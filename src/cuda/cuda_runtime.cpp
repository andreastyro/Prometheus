// The CUDA driver and NVRTC, loaded at runtime. See ml/cuda.hpp.
//
// Only the handful of driver API entry points used are declared, by hand, with
// the ABI types they have in cuda.h; nothing here includes a CUDA header.
#include "ml/cuda.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
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
    #include <sys/stat.h>
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
    // optional: the kernel profiler's GPU timestamps
    CUresult (*cuEventCreate)(void**, unsigned);
    CUresult (*cuEventRecord)(void*, CUstream);
    CUresult (*cuEventElapsedTime)(float*, void*, void*);
    CUresult (*cuEventDestroy)(void*);
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
    int (*nvrtcVersion)(int*, int*);                 // optional: part of the disk cache key
    int (*nvrtcGetCUBINSize)(nvrtcProgram, size_t*); // optional (CUDA 11.1+): machine code
    int (*nvrtcGetCUBIN)(nvrtcProgram, char*);
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
// one module per source text: a source holding many kernels is compiled once
unordered_map<const char*, CUmodule> g_modules;
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
        (bind(lib, d.cuEventCreate, "cuEventCreate") || true) && (bind(lib, d.cuEventRecord, "cuEventRecord") || true) &&
        (bind(lib, d.cuEventElapsedTime, "cuEventElapsedTime") || true) &&
        (bind(lib, d.cuEventDestroy, "cuEventDestroy_v2") || true) &&
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
    bind(lib, r.nvrtcVersion, "nvrtcVersion");
    if (!bind(lib, r.nvrtcGetCUBINSize, "nvrtcGetCUBINSize") || !bind(lib, r.nvrtcGetCUBIN, "nvrtcGetCUBIN"))
        r.nvrtcGetCUBIN = nullptr;

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

// ── the kernel disk cache ────────────────────────────────────────────────────
//
// Each source is compiled once per process, all its kernels in one module.
// NVRTC compiles straight to this GPU's machine code (a cubin) when it can:
// loading PTX made the driver translate it on every first use -- 25-35 ms per
// source, most of a cold start, and its own cache did not remove it. The
// result is kept on disk, keyed by a hash of the source, the target and the
// NVRTC version, so a later process neither compiles nor translates.
// $PROMETHEUS_KERNEL_CACHE picks the folder, or "0" turns the cache off; by
// default %LOCALAPPDATA%\prometheus\kernels or ~/.cache/prometheus/kernels.

uint64_t fnv1a(const string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

string cache_dir() {
    static const string dir = [] {
        string env = getenv("PROMETHEUS_KERNEL_CACHE") ? getenv("PROMETHEUS_KERNEL_CACHE") : "";
        // trimmed: cmd's `set X= ` leaves a space behind
        env.erase(0, env.find_first_not_of(" \t"));
        env.erase(env.find_last_not_of(" \t") + 1);
        if (env == "0") return string();
        if (!env.empty()) return env;
#ifdef _WIN32
        const char* base = getenv("LOCALAPPDATA");
        return base ? string(base) + "\\prometheus\\kernels" : string();
#else
        if (const char* x = getenv("XDG_CACHE_HOME")) if (*x) return string(x) + "/prometheus/kernels";
        const char* home = getenv("HOME");
        return home ? string(home) + "/.cache/prometheus/kernels" : string();
#endif
    }();
    return dir;
}

// machine code for this GPU: NVRTC can emit it, and its newest target shares
// the GPU's major version (sm_XY code runs on X.Y and later X.*)
bool use_cubin() { return g.r.nvrtcGetCUBIN && g.arch / 10 == g.cc / 10; }

string cache_path(const char* source) {
    const string dir = cache_dir();
    if (dir.empty()) return {};
    int major = 0, minor = 0;
    if (g.r.nvrtcVersion) g.r.nvrtcVersion(&major, &minor);
    // the leading tag changes whenever the compile options below do
    const bool bin = use_cubin();
    const string key = string(bin ? "v2 sm_" : "v2 compute_") + to_string(g.arch) + " nvrtc " + to_string(major) +
                       "." + to_string(minor) + "\n" + source;
    char name[48];
    std::snprintf(name, sizeof name, "%016llx.%s", (unsigned long long)fnv1a(key), bin ? "cubin" : "ptx");
    return dir + "/" + name;
}

string read_file(const string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// every folder on the way to `dir` (std::filesystem would raise the macOS
// deployment target the wheels build for)
void make_dirs(const string& dir) {
    for (size_t i = 1; i <= dir.size(); i++) {
        if (i < dir.size() && dir[i] != '/' && dir[i] != '\\') continue;
        const string part = dir.substr(0, i);
#ifdef _WIN32
        CreateDirectoryA(part.c_str(), nullptr);
#else
        mkdir(part.c_str(), 0755);
#endif
    }
}

// written beside its final name and renamed into place, so a reader never
// sees half a file and two processes writing at once both end up correct
void write_file(const string& path, const string& data) {
    make_dirs(path.substr(0, path.find_last_of("/\\")));
    const string tmp = path + "." + to_string(std::hash<std::thread::id>()(std::this_thread::get_id())) + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out) return;
        out.write(data.data(), (std::streamsize)data.size());
        if (!out) { out.close(); std::remove(tmp.c_str()); return; }
    }
    // replacing what is there: a damaged file is rewritten, and a racing
    // process wrote the same bytes (std::rename will not replace on Windows)
#ifdef _WIN32
    const bool moved = MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    const bool moved = std::rename(tmp.c_str(), path.c_str()) == 0;
#endif
    if (!moved) std::remove(tmp.c_str());
}

// the module image for a source -- a cubin, or PTX -- from the disk cache, or
// compiled by NVRTC (and cached)
string ptx_for(const char* source, const char* name, bool use_cache) {
    {
        lock_guard<mutex> lock(g_mu);
        init_nvrtc_locked();
        if (!g.nvrtc_ok) throw runtime_error("cuda: cannot compile kernels: " + g.nvrtc_why);
    }
    const string path = use_cache ? cache_path(source) : string();
    if (!path.empty()) {
        string ptx = read_file(path);
        if (!ptx.empty()) return ptx;
    }
    const Nvrtc& r = g.r;
    nvrtcProgram prog = nullptr;
    int rc = r.nvrtcCreateProgram(&prog, source, name, 0, nullptr, nullptr);
    if (rc != 0) throw runtime_error(string("cuda: nvrtcCreateProgram: ") + r.nvrtcGetErrorString(rc));
    const bool bin = use_cubin();
    const string arch = (bin ? "--gpu-architecture=sm_" : "--gpu-architecture=compute_") + to_string(g.arch);
    const char* opts[] = {arch.c_str(), "--std=c++14", "-default-device"};
    rc = r.nvrtcCompileProgram(prog, 3, opts);
    if (rc != 0) {
        size_t n = 0;
        r.nvrtcGetProgramLogSize(prog, &n);
        string log(n, '\0');
        if (n) r.nvrtcGetProgramLog(prog, &log[0]);
        r.nvrtcDestroyProgram(&prog);
        throw runtime_error(string("cuda: compiling kernel '") + name + "' failed:\n" + log);
    }
    size_t n = 0;
    string ptx;
    if (bin) {
        r.nvrtcGetCUBINSize(prog, &n);
        ptx.assign(n, '\0');
        r.nvrtcGetCUBIN(prog, &ptx[0]);
    } else {
        r.nvrtcGetPTXSize(prog, &n);
        ptx.assign(n, '\0');
        r.nvrtcGetPTX(prog, &ptx[0]);
        while (!ptx.empty() && ptx.back() == '\0') ptx.pop_back();
    }
    r.nvrtcDestroyProgram(&prog);
    if (!path.empty()) write_file(path, ptx);
    return ptx;
}

// the module holding every kernel of `source`, compiled (or read) once
CUmodule module_for(const Kernel& k) {
    {
        lock_guard<mutex> lock(g_kernel_mu);
        auto it = g_modules.find(k.source);
        if (it != g_modules.end()) return it->second;
    }
    CUmodule mod = nullptr;
    string ptx = ptx_for(k.source, k.name, true);
    if (g.d.cuModuleLoadData(&mod, ptx.c_str()) != 0) {
        // a damaged cache file: compile afresh (which rewrites it)
        ptx = ptx_for(k.source, k.name, false);
        const string path = cache_path(k.source);
        if (!path.empty()) write_file(path, ptx);
        check(g.d.cuModuleLoadData(&mod, ptx.c_str()), "cuModuleLoadData");
    }
    lock_guard<mutex> lock(g_kernel_mu);
    auto it = g_modules.find(k.source);
    if (it != g_modules.end()) return it->second;    // another thread got there first
    g_modules[k.source] = mod;
    return mod;
}

CUfunction function(const Kernel& k) {
    bind_context();
    const KernelKey key{k.name, k.source};
    {
        lock_guard<mutex> lock(g_kernel_mu);
        auto it = g_kernels.find(key);
        if (it != g_kernels.end()) return it->second;
    }
    CUmodule mod = module_for(k);
    CUfunction fn = nullptr;
    check(g.d.cuModuleGetFunction(&fn, mod, k.name), "cuModuleGetFunction");
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
    ProfScope p(profiling() ? string("sgemm ") + (ta ? "T" : "N") + (tb ? "T" : "N") + " " + to_string(m) + "x" + to_string(n) + "x" + to_string(k) : string());
    check_blas(g_blas.sgemm(h, tb ? 1 : 0, ta ? 1 : 0, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc), "sgemm");
}

void sgemm_batched(bool ta, bool tb, int m, int n, int k, float alpha, const float* a, int lda,
                   long long sa, const float* b, int ldb, long long sb, float beta, float* c, int ldc,
                   long long sc, int batch) {
    if (m == 0 || n == 0 || batch == 0) return;
    auto h = blas_handle();
    ProfScope p(profiling() ? string("sgemm_batched ") + (ta ? "T" : "N") + (tb ? "T" : "N") + " " + to_string(m) + "x" + to_string(n) + "x" + to_string(k) + " *" + to_string(batch) : string());
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
    ProfScope p(profiling() ? string("gemm_half ") + (ta ? "T" : "N") + (tb ? "T" : "N") + " " + to_string(m) + "x" + to_string(n) + "x" + to_string(k) + " *" + to_string(batch) : string());
    check_blas(g_blas.gemm_ex(h, tb ? 1 : 0, ta ? 1 : 0, n, m, k, &alpha, b, type, ldb, sb, a, type, lda, sa,
                              &beta, c, 0, ldc, sc, batch, 68, -1), "GemmEx");
}

// ── cuDNN ────────────────────────────────────────────────────────────────────
//
// Only the legacy convolution API is used (still present in cuDNN 9): tensor,
// filter and convolution descriptors, the _v7 heuristics to pick an algorithm,
// and the three convolution calls. Each distinct shape gets a plan -- its
// descriptors, chosen algorithms and workspace sizes -- made once and kept.

namespace {
using dnnHandle = void*;
using dnnDesc = void*;
// cudnnConvolution{Fwd,BwdData,BwdFilter}AlgoPerf_t share this layout
struct DnnPerf {
    int algo;
    int status;
    float time;
    size_t memory;
    int determinism;
    int math_type;
    int reserved[3];
};
struct Dnn {
    size_t (*get_version)();
    const char* (*error_string)(int);
    int (*create)(dnnHandle*);
    int (*set_stream)(dnnHandle, void*);
    int (*create_tensor)(dnnDesc*);
    int (*set_tensor4d)(dnnDesc, int, int, int, int, int, int);
    int (*create_filter)(dnnDesc*);
    int (*set_filter4d)(dnnDesc, int, int, int, int, int, int);
    int (*create_conv)(dnnDesc*);
    int (*set_conv2d)(dnnDesc, int, int, int, int, int, int, int, int);
    int (*fwd_algo)(dnnHandle, dnnDesc, dnnDesc, dnnDesc, dnnDesc, int, int*, DnnPerf*);
    int (*fwd_ws)(dnnHandle, dnnDesc, dnnDesc, dnnDesc, dnnDesc, int, size_t*);
    int (*fwd)(dnnHandle, const void*, dnnDesc, const void*, dnnDesc, const void*, dnnDesc, int, void*, size_t,
               const void*, dnnDesc, void*);
    int (*bwd_data_algo)(dnnHandle, dnnDesc, dnnDesc, dnnDesc, dnnDesc, int, int*, DnnPerf*);
    int (*bwd_data_ws)(dnnHandle, dnnDesc, dnnDesc, dnnDesc, dnnDesc, int, size_t*);
    int (*bwd_data)(dnnHandle, const void*, dnnDesc, const void*, dnnDesc, const void*, dnnDesc, int, void*, size_t,
                    const void*, dnnDesc, void*);
    int (*bwd_filter_algo)(dnnHandle, dnnDesc, dnnDesc, dnnDesc, dnnDesc, int, int*, DnnPerf*);
    int (*bwd_filter_ws)(dnnHandle, dnnDesc, dnnDesc, dnnDesc, dnnDesc, int, size_t*);
    int (*bwd_filter)(dnnHandle, const void*, dnnDesc, const void*, dnnDesc, const void*, dnnDesc, int, void*, size_t,
                      const void*, dnnDesc, void*);
    // optional: time every algorithm on the real shape (cudnnFind*); same
    // arguments as the _v7 heuristics
    int (*find_fwd)(dnnHandle, dnnDesc, dnnDesc, dnnDesc, dnnDesc, int, int*, DnnPerf*);
    int (*find_bwd_data)(dnnHandle, dnnDesc, dnnDesc, dnnDesc, dnnDesc, int, int*, DnnPerf*);
    int (*find_bwd_filter)(dnnHandle, dnnDesc, dnnDesc, dnnDesc, dnnDesc, int, int*, DnnPerf*);
};
Dnn       g_dnn{};
dnnHandle g_dnn_handle = nullptr;
bool      g_dnn_tried = false, g_dnn_ok = false;
string    g_dnn_why;
size_t    g_dnn_version = 0;

struct DnnPlan {
    dnnDesc x, w, y, conv;
    int fwd_algo, data_algo, filter_algo;
    size_t fwd_ws, data_ws, filter_ws;
};
map<std::tuple<int, int, int, int, int, int, int, int>, DnnPlan> g_dnn_plans;
mutex g_dnn_mu;

// 91002 -> "9.10.2"
string dnn_version_string() {
    return to_string(g_dnn_version / 10000) + "." + to_string(g_dnn_version / 100 % 100) + "." +
           to_string(g_dnn_version % 100);
}

// a 3x3 convolution of a 4x4 image, on raw device memory (the caching
// allocator would take g_mu, which the caller holds); true if it ran
bool dnn_probe() {
    const Dnn& d = g_dnn;
    dnnDesc x = nullptr, w = nullptr, y = nullptr, conv = nullptr;
    CUdeviceptr px = 0, pw = 0, py = 0, pws = 0;
    bool ok = d.create_tensor(&x) == 0 && d.set_tensor4d(x, 0, 0, 1, 1, 4, 4) == 0 &&
              d.create_tensor(&y) == 0 && d.set_tensor4d(y, 0, 0, 1, 1, 2, 2) == 0 &&
              d.create_filter(&w) == 0 && d.set_filter4d(w, 0, 0, 1, 1, 3, 3) == 0 &&
              d.create_conv(&conv) == 0 && d.set_conv2d(conv, 0, 0, 1, 1, 1, 1, 1, 0) == 0;
    DnnPerf perf[16];
    int n = 0, algo = 0;
    size_t ws = 0;
    ok = ok && d.fwd_algo(g_dnn_handle, x, w, conv, y, 16, &n, perf) == 0 && n > 0;
    if (ok) algo = perf[0].algo;
    ok = ok && d.fwd_ws(g_dnn_handle, x, w, conv, y, algo, &ws) == 0;
    ok = ok && g.d.cuMemAlloc(&px, 64) == 0 && g.d.cuMemAlloc(&pw, 64) == 0 && g.d.cuMemAlloc(&py, 64) == 0 &&
         (ws == 0 || g.d.cuMemAlloc(&pws, ws) == 0);
    const float one = 1.0f, zero = 0.0f;
    ok = ok && d.fwd(g_dnn_handle, &one, x, (void*)px, w, (void*)pw, conv, algo, (void*)pws, ws, &zero, y,
                     (void*)py) == 0;
    ok = ok && g.d.cuStreamSynchronize(g.stream) == 0;
    for (CUdeviceptr p : {px, pw, py, pws}) if (p) g.d.cuMemFree(p);
    return ok;  // descriptors are a few bytes; not worth destroy entry points
}

void init_dnn_locked() {
    if (g_dnn_tried) return;
    g_dnn_tried = true;
    init_locked();
    if (!g.ok) { g_dnn_why = g.why; return; }
    if (const char* off = getenv("PROMETHEUS_CUDNN"))
        if (string(off) == "0") { g_dnn_why = "turned off by PROMETHEUS_CUDNN=0"; return; }
#ifdef _WIN32
    void* lib = open_first({"cudnn64_9.dll", "cudnn64_8.dll"});
#else
    void* lib = open_first({"libcudnn.so.9", "libcudnn.so.8", "libcudnn.so"});
#endif
    if (!lib) { g_dnn_why = "not installed (pip install nvidia-cudnn-cu12 for faster convolutions)"; return; }
    Dnn& d = g_dnn;
    const bool all =
        bind(lib, d.get_version, "cudnnGetVersion") && bind(lib, d.error_string, "cudnnGetErrorString") &&
        bind(lib, d.create, "cudnnCreate") && bind(lib, d.set_stream, "cudnnSetStream") &&
        bind(lib, d.create_tensor, "cudnnCreateTensorDescriptor") && bind(lib, d.set_tensor4d, "cudnnSetTensor4dDescriptor") &&
        bind(lib, d.create_filter, "cudnnCreateFilterDescriptor") && bind(lib, d.set_filter4d, "cudnnSetFilter4dDescriptor") &&
        bind(lib, d.create_conv, "cudnnCreateConvolutionDescriptor") &&
        bind(lib, d.set_conv2d, "cudnnSetConvolution2dDescriptor") &&
        bind(lib, d.fwd_algo, "cudnnGetConvolutionForwardAlgorithm_v7") &&
        bind(lib, d.fwd_ws, "cudnnGetConvolutionForwardWorkspaceSize") && bind(lib, d.fwd, "cudnnConvolutionForward") &&
        bind(lib, d.bwd_data_algo, "cudnnGetConvolutionBackwardDataAlgorithm_v7") &&
        bind(lib, d.bwd_data_ws, "cudnnGetConvolutionBackwardDataWorkspaceSize") &&
        bind(lib, d.bwd_data, "cudnnConvolutionBackwardData") &&
        bind(lib, d.bwd_filter_algo, "cudnnGetConvolutionBackwardFilterAlgorithm_v7") &&
        bind(lib, d.bwd_filter_ws, "cudnnGetConvolutionBackwardFilterWorkspaceSize") &&
        bind(lib, d.bwd_filter, "cudnnConvolutionBackwardFilter");
    if (!all) { g_dnn_why = "cuDNN is missing a convolution entry point"; return; }
    const char* bench = getenv("PROMETHEUS_CUDNN_BENCHMARK");
    if (!(bench && string(bench) == "0") && !(bind(lib, d.find_fwd, "cudnnFindConvolutionForwardAlgorithm") &&
                                              bind(lib, d.find_bwd_data, "cudnnFindConvolutionBackwardDataAlgorithm") &&
                                              bind(lib, d.find_bwd_filter, "cudnnFindConvolutionBackwardFilterAlgorithm")))
        d.find_fwd = nullptr;
    g_dnn_version = d.get_version();
    if (g.d.cuCtxSetCurrent(g.ctx) != 0 || d.create(&g_dnn_handle) != 0 || d.set_stream(g_dnn_handle, g.stream) != 0) {
        g_dnn_why = "cudnnCreate failed (this cuDNN may not support compute " + to_string(g.cc / 10) + "." +
                    to_string(g.cc % 10) + ")";
        return;
    }
    // Newer cuDNN releases load and plan on older GPUs and then fail at the
    // first kernel (9.28 on a GTX 970: EXECUTION_FAILED_CUDART), so one tiny
    // convolution is run now and the built-in kernels are kept if it fails.
    if (!dnn_probe()) {
        g_dnn_why = "cuDNN " + dnn_version_string() + " cannot run on compute " + to_string(g.cc / 10) + "." +
                    to_string(g.cc % 10) + " (cuDNN 9.10, as bundled with PyTorch, can)";
        return;
    }
    g_dnn_ok = true;
}

void check_dnn(int status, const char* what) {
    if (status != 0)
        throw runtime_error(string("cuDNN ") + what + " failed: " +
                            (g_dnn.error_string ? g_dnn.error_string(status) : to_string(status)));
}

// the plan for a shape: descriptors, an algorithm for each direction and its
// workspace size. Each algorithm is timed on this shape once (what PyTorch's
// cudnn.benchmark does) and the fastest kept; with PROMETHEUS_CUDNN_BENCHMARK=0
// the heuristic's first choice is taken untimed.
DnnPlan& dnn_plan(const ConvShape& s) {
    lock_guard<mutex> lock(g_dnn_mu);
    auto key = std::make_tuple(s.B, s.C, s.H, s.W, s.F, s.K, s.stride, s.pad);
    auto it = g_dnn_plans.find(key);
    if (it != g_dnn_plans.end()) return it->second;
    const Dnn& d = g_dnn;
    const int Ho = (s.H + 2 * s.pad - s.K) / s.stride + 1, Wo = (s.W + 2 * s.pad - s.K) / s.stride + 1;
    DnnPlan p{};
    // NCHW = 0, float = 0, cross-correlation = 1
    check_dnn(d.create_tensor(&p.x), "tensor descriptor");
    check_dnn(d.set_tensor4d(p.x, 0, 0, s.B, s.C, s.H, s.W), "tensor descriptor");
    check_dnn(d.create_tensor(&p.y), "tensor descriptor");
    check_dnn(d.set_tensor4d(p.y, 0, 0, s.B, s.F, Ho, Wo), "tensor descriptor");
    check_dnn(d.create_filter(&p.w), "filter descriptor");
    check_dnn(d.set_filter4d(p.w, 0, 0, s.F, s.C, s.K, s.K), "filter descriptor");
    check_dnn(d.create_conv(&p.conv), "convolution descriptor");
    check_dnn(d.set_conv2d(p.conv, s.pad, s.pad, s.stride, s.stride, 1, 1, 1, 0), "convolution descriptor");
    // results come fastest first; skip failures and workspaces over 256 MB
    auto pick = [](const DnnPerf* perf, int n) {
        for (int i = 0; i < n; i++)
            if (perf[i].status == 0 && perf[i].memory <= (size_t(256) << 20)) return perf[i].algo;
        return perf[0].algo;
    };
    const bool timed = d.find_fwd != nullptr;
    // cudnnFind runs synchronously on the default stream: let ours drain first
    if (timed) check(g.d.cuStreamSynchronize(g.stream), "cuStreamSynchronize");
    DnnPerf perf[16];
    int n = 0;
    check_dnn((timed ? d.find_fwd : d.fwd_algo)(g_dnn_handle, p.x, p.w, p.conv, p.y, 16, &n, perf),
              "forward algorithm");
    p.fwd_algo = pick(perf, n);
    check_dnn(d.fwd_ws(g_dnn_handle, p.x, p.w, p.conv, p.y, p.fwd_algo, &p.fwd_ws), "forward workspace");
    check_dnn((timed ? d.find_bwd_data : d.bwd_data_algo)(g_dnn_handle, p.w, p.y, p.conv, p.x, 16, &n, perf),
              "backward-data algorithm");
    p.data_algo = pick(perf, n);
    check_dnn(d.bwd_data_ws(g_dnn_handle, p.w, p.y, p.conv, p.x, p.data_algo, &p.data_ws), "backward-data workspace");
    check_dnn((timed ? d.find_bwd_filter : d.bwd_filter_algo)(g_dnn_handle, p.x, p.y, p.conv, p.w, 16, &n, perf),
              "backward-filter algorithm");
    p.filter_algo = pick(perf, n);
    check_dnn(d.bwd_filter_ws(g_dnn_handle, p.x, p.y, p.conv, p.w, p.filter_algo, &p.filter_ws),
              "backward-filter workspace");
    return g_dnn_plans.emplace(key, p).first->second;
}

// a workspace from the caching allocator for the length of one call: the
// stream orders the reuse, so it can go straight back
struct Workspace {
    void* p = nullptr;
    size_t n = 0;
    explicit Workspace(size_t bytes) : n(bytes) { if (n) p = alloc(n); }
    ~Workspace() { if (p) cuda::free(p, n); }
};
} // namespace

bool dnn_available() {
    if (!available()) return false;
    lock_guard<mutex> lock(g_mu);
    init_dnn_locked();
    return g_dnn_ok;
}

string dnn_status() {
    if (!available()) return "unavailable: " + state().why;
    lock_guard<mutex> lock(g_mu);
    init_dnn_locked();
    if (!g_dnn_ok) return "not used: " + g_dnn_why;
    return "cuDNN " + dnn_version_string();
}

void dnn_conv_forward(const ConvShape& s, const float* x, const float* w, float* y) {
    bind_context();
    DnnPlan& p = dnn_plan(s);
    Workspace ws(p.fwd_ws);
    const float one = 1.0f, zero = 0.0f;
    ProfScope ps("cudnn conv forward");
    check_dnn(g_dnn.fwd(g_dnn_handle, &one, p.x, x, p.w, w, p.conv, p.fwd_algo, ws.p, ws.n, &zero, p.y, y),
              "convolution forward");
}

void dnn_conv_backward_data(const ConvShape& s, const float* w, const float* dy, float* dx) {
    bind_context();
    DnnPlan& p = dnn_plan(s);
    Workspace ws(p.data_ws);
    const float one = 1.0f, zero = 0.0f;
    ProfScope ps("cudnn conv backward data");
    check_dnn(g_dnn.bwd_data(g_dnn_handle, &one, p.w, w, p.y, dy, p.conv, p.data_algo, ws.p, ws.n, &zero, p.x, dx),
              "convolution backward data");
}

void dnn_conv_backward_filter(const ConvShape& s, const float* x, const float* dy, float* dw) {
    bind_context();
    DnnPlan& p = dnn_plan(s);
    Workspace ws(p.filter_ws);
    const float one = 1.0f, zero = 0.0f;
    ProfScope ps("cudnn conv backward filter");
    check_dnn(g_dnn.bwd_filter(g_dnn_handle, &one, p.x, x, p.y, dy, p.conv, p.filter_algo, ws.p, ws.n, &zero, p.w, dw),
              "convolution backward filter");
}

// ── kernel profiler ──────────────────────────────────────────────────────────
//
// With profiling on, every kernel launch, cuBLAS call and cuDNN call is
// bracketed by a pair of GPU events on the stream, so its time is what the
// GPU spent on it -- no synchronising per op, which on Windows inflates every
// small op by the driver's submission latency. The report sums by name.

namespace {
struct ProfRecord { string name; void* a; void* b; };
vector<ProfRecord> g_prof;
vector<void*>      g_event_pool;
mutex              g_prof_mu;
bool               g_prof_on = [] {
    const char* e = getenv("PROMETHEUS_PROFILE_KERNELS");
    return e && *e && string(e) != "0";
}();

void* take_event() {
    if (!g_event_pool.empty()) { void* e = g_event_pool.back(); g_event_pool.pop_back(); return e; }
    void* e = nullptr;
    if (g.d.cuEventCreate(&e, 0) != 0) return nullptr;
    return e;
}
} // namespace

ProfScope::ProfScope(const char* name) {
    if (!g_prof_on || !g.d.cuEventRecord) return;
    lock_guard<mutex> lock(g_prof_mu);
    a_ = take_event();
    if (!a_) return;
    name_ = name;
    g.d.cuEventRecord(a_, g.stream);
}
ProfScope::ProfScope(const string& name) : ProfScope(name.c_str()) {}
ProfScope::~ProfScope() {
    if (!a_) return;
    lock_guard<mutex> lock(g_prof_mu);
    void* b = take_event();
    if (!b) { g_event_pool.push_back(a_); return; }
    g.d.cuEventRecord(b, g.stream);
    g_prof.push_back({name_, a_, b});
}

void profile(bool on) {
    lock_guard<mutex> lock(g_prof_mu);
    g_prof_on = on;
}
bool profiling() { return g_prof_on; }

void profile_reset() {
    if (available()) synchronize();
    lock_guard<mutex> lock(g_prof_mu);
    for (auto& r : g_prof) { g_event_pool.push_back(r.a); g_event_pool.push_back(r.b); }
    g_prof.clear();
}

vector<ProfEntry> profile_entries() {
    if (available()) synchronize();
    lock_guard<mutex> lock(g_prof_mu);
    std::map<string, ProfEntry> by;
    for (auto& r : g_prof) {
        float ms = 0.0f;
        if (g.d.cuEventElapsedTime(&ms, r.a, r.b) != 0) continue;
        auto& e = by[r.name];
        e.name = r.name;
        e.ms += ms;
        e.calls++;
    }
    vector<ProfEntry> out;
    for (auto& [k, v] : by) out.push_back(v);
    std::sort(out.begin(), out.end(), [](const ProfEntry& x, const ProfEntry& y) { return x.ms > y.ms; });
    return out;
}

string profile_report(int top) {
    auto rows = profile_entries();
    double total = 0.0;
    for (auto& r : rows) total += r.ms;
    string out;
    char line[256];
    std::snprintf(line, sizeof line, "GPU time %.2f ms over %zu kinds\n", total, rows.size());
    out += line;
    int k = 0;
    for (auto& r : rows) {
        if (top > 0 && k++ >= top) break;
        std::snprintf(line, sizeof line, "  %-44s %9.3f ms %5.1f%% %6d calls\n", r.name.c_str(), r.ms,
                      total > 0 ? 100.0 * r.ms / total : 0.0, r.calls);
        out += line;
    }
    return out;
}

void launch(const Kernel& k, int64_t n, void** args, unsigned block) {
    if (n <= 0) return;
    const int64_t grid = (n + block - 1) / block;
    launch(k, (unsigned)min<int64_t>(grid, 0x7fffffff), 1, 1, block, 1, 1, 0, args);
}

void launch(const Kernel& k, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
            unsigned bz, unsigned shared, void** args) {
    CUfunction fn = function(k);
    ProfScope p(k.name);
    check(g.d.cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, shared, g.stream, args, nullptr),
          (string("launching ") + k.name).c_str());
}

} // namespace cuda
