#include "ml/blas.hpp"
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

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

namespace blas {
namespace {

// Standard CBLAS enum values. Every implementation uses these numbers, which
// is what lets the call go through without a cblas.h.
constexpr int RowMajor = 101, NoTrans = 111, Trans = 112;

// When nothing is asked for, try MKL first: measured on a Zen 5 laptop it ran
// the GPT's matmuls faster than OpenBLAS 0.3.26, and it is what PyTorch uses.
const char* const AUTO_ORDER[] = {"mkl", "openblas"};

using sgemm_t       = void (*)(int, int, int, int, int, int, float, const float*, int,
                               const float*, int, float, float*, int);
using set_threads_t = void (*)(int);
using get_threads_t = int (*)();

struct Backend {
    std::string   name = "tiled";
    std::string   path;
    sgemm_t       sgemm       = nullptr;
    set_threads_t set_threads = nullptr;
    get_threads_t get_threads = nullptr;
};

Backend                  g_backend;
std::vector<std::string> g_dirs;
std::mutex               g_mu;
std::atomic<bool>        g_ready{false};

std::vector<std::string> file_names(const std::string& lib) {
#if defined(_WIN32)
    if (lib == "openblas") return {"libopenblas.dll", "openblas.dll"};
    if (lib == "mkl")      return {"mkl_rt.3.dll", "mkl_rt.2.dll", "mkl_rt.dll"};
#elif defined(__APPLE__)
    if (lib == "openblas") return {"libopenblas.0.dylib", "libopenblas.dylib"};
    if (lib == "mkl")      return {"libmkl_rt.2.dylib", "libmkl_rt.dylib"};
#else
    if (lib == "openblas") return {"libopenblas.so.0", "libopenblas.so"};
    if (lib == "mkl")      return {"libmkl_rt.so.2", "libmkl_rt.so.1", "libmkl_rt.so"};
#endif
    return {};
}

#ifdef _WIN32
// MKL's runtime loads its core, threading layer and Intel's OpenMP runtime
// itself, by name, after mkl_rt is open. Those live beside mkl_rt, so that
// directory has to be on PATH for its own LoadLibrary calls to find them.
void prepend_to_path(const std::string& dir) {
    std::vector<char> buf(32768);
    DWORD n = GetEnvironmentVariableA("PATH", buf.data(), (DWORD)buf.size());
    std::string path(buf.data(), n < buf.size() ? n : 0);
    if (path.find(dir) != std::string::npos) return;
    SetEnvironmentVariableA("PATH", (dir + ";" + path).c_str());
}
#endif

void* open_library(const std::string& file) {
#ifdef _WIN32
    const auto slash = file.find_last_of("\\/");
    if (slash != std::string::npos) {
        prepend_to_path(file.substr(0, slash));
        return (void*)LoadLibraryExA(file.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
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

std::string loaded_path(void* lib, void* any_symbol) {
#ifdef _WIN32
    (void)any_symbol;
    char buf[MAX_PATH * 4];
    DWORD n = GetModuleFileNameA((HMODULE)lib, buf, sizeof buf);
    return std::string(buf, n);
#else
    (void)lib;
    Dl_info info;
    if (dladdr(any_symbol, &info) && info.dli_fname) return info.dli_fname;
    return "";
#endif
}

// Directory holding this code: the .pyd / .so for the Python module, the
// executable for a C++ program that links the library statically. A BLAS
// copied next to either is found without any configuration.
std::string own_directory() {
#ifdef _WIN32
    HMODULE self = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR)&own_directory, &self))
        return "";
    char buf[MAX_PATH * 4];
    std::string p(buf, GetModuleFileNameA(self, buf, sizeof buf));
#else
    Dl_info info;
    if (!dladdr((void*)&own_directory, &info) || !info.dli_fname) return "";
    std::string p = info.dli_fname;
#endif
    const auto slash = p.find_last_of("\\/");
    return slash == std::string::npos ? "" : p.substr(0, slash);
}

// Libraries are never unloaded: both keep thread pools alive, and freeing
// one under a pool that is still spinning crashes the process.
bool try_load(const std::string& lib, const std::string& path, Backend* out) {
    std::vector<std::string> tries;
    if (!path.empty()) {
        tries.push_back(path);
    } else {
        std::vector<std::string> dirs = g_dirs;
        const std::string own = own_directory();
        if (!own.empty()) dirs.insert(dirs.begin(), own);
        for (const auto& d : dirs)
            for (const auto& f : file_names(lib))
                tries.push_back(d + "/" + f);
        for (const auto& f : file_names(lib))
            tries.push_back(f);                  // the system's own search
    }

    for (const auto& file : tries) {
        void* h = open_library(file);
        if (!h) continue;
        auto sgemm = (sgemm_t)symbol(h, "cblas_sgemm");
        if (!sgemm) continue;

        Backend b;
        b.name  = lib;
        b.sgemm = sgemm;
        b.path  = loaded_path(h, (void*)sgemm);
        if (lib == "openblas") {
            b.set_threads = (set_threads_t)symbol(h, "openblas_set_num_threads");
            b.get_threads = (get_threads_t)symbol(h, "openblas_get_num_threads");
        } else {
            b.set_threads = (set_threads_t)symbol(h, "MKL_Set_Num_Threads");
            if (!b.set_threads) b.set_threads = (set_threads_t)symbol(h, "mkl_set_num_threads");
            b.get_threads = (get_threads_t)symbol(h, "MKL_Get_Max_Threads");
            if (!b.get_threads) b.get_threads = (get_threads_t)symbol(h, "mkl_get_max_threads");
        }
        *out = b;
        return true;
    }
    return false;
}

bool select_locked(const std::string& which, const std::string& path, std::string* error) {
    if (which == "tiled") {
        g_backend = Backend{};
        return true;
    }
    if (which == "auto") {
        for (const char* lib : AUTO_ORDER) {
            Backend b;
            if (try_load(lib, "", &b)) { g_backend = b; return true; }
        }
        g_backend = Backend{};
        return true;
    }
    if (which == "openblas" || which == "mkl") {
        Backend b;
        if (try_load(which, path, &b)) { g_backend = b; return true; }
        if (error)
            *error = "could not load " + which + (path.empty() ? "" : " from " + path) +
                     ": no library with cblas_sgemm found. Pass its path, or set "
                     "PROMETHEUS_BLAS_PATH.";
        return false;
    }
    if (error)
        *error = "unknown BLAS '" + which + "'; expected auto, openblas, mkl or tiled";
    return false;
}

// The startup choice, made from the environment on first use. An explicit
// request that cannot be met throws and leaves the flag unset, so every
// later call throws too -- until set_blas() makes a choice that works.
void ensure_init() {
    if (g_ready.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_ready.load(std::memory_order_relaxed)) return;
    const char* w = std::getenv("PROMETHEUS_BLAS");
    const char* p = std::getenv("PROMETHEUS_BLAS_PATH");
    const std::string which = (w && *w) ? w : "auto";
    std::string err;
    if (!select_locked(which, p ? p : "", &err))
        throw std::runtime_error("PROMETHEUS_BLAS=" + which + ": " + err);
    g_ready.store(true, std::memory_order_release);
}

} // namespace

bool available()                  { ensure_init(); return g_backend.sgemm != nullptr; }
const std::string& name()         { ensure_init(); return g_backend.name; }
const std::string& library_path() { ensure_init(); return g_backend.path; }
bool thread_control()             { ensure_init(); return g_backend.set_threads != nullptr; }

int get_threads() {
    ensure_init();
    return g_backend.get_threads ? g_backend.get_threads() : 0;
}

void set_threads(int n) {
    ensure_init();
    if (g_backend.set_threads) g_backend.set_threads(n);
}

void sgemm(bool trans_a, bool trans_b, int m, int n, int k,
           float alpha, const float* a, int lda, const float* b, int ldb,
           float beta, float* c, int ldc) {
    g_backend.sgemm(RowMajor, trans_a ? Trans : NoTrans, trans_b ? Trans : NoTrans,
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void set_search_dirs(const std::vector<std::string>& dirs) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_dirs = dirs;
}

bool select(const std::string& which, const std::string& path, std::string* error) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!select_locked(which, path, error)) return false;
    g_ready.store(true, std::memory_order_release);   // overrides the environment
    return true;
}

} // namespace blas
