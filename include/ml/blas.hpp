#pragma once
#include <string>
#include <vector>

/// Runtime BLAS selection.
///
/// OpenBLAS and Intel MKL both export the standard `cblas_sgemm`, so instead
/// of linking one of them at build time, the library loads whichever is
/// chosen when it is first needed (LoadLibrary / dlopen) and calls through a
/// function pointer. One build can then use either, or neither, and switching
/// costs no rebuild. Nothing needs BLAS headers to compile.
///
/// Selection, in order of precedence:
///   - prometheus.set_blas("mkl" | "openblas" | "tiled" | "auto", path=None)
///   - the PROMETHEUS_BLAS environment variable, read once at first use, with
///     PROMETHEUS_BLAS_PATH naming the exact library file if needed
///   - "auto": try each library in turn, fall back to the portable tiled
///     kernel if none loads
///
/// Asking for a specific library that cannot be loaded is an error rather
/// than a silent fallback: a benchmark run against the wrong backend is
/// exactly the mistake backend() exists to catch.
namespace blas {

/// True when a BLAS library is loaded and in use.
bool available();

/// "openblas", "mkl" or "tiled".
const std::string& name();

/// Full path of the loaded library, empty for "tiled".
const std::string& library_path();

/// C = alpha * op(A) @ op(B) + beta * C, row-major. Only call when available().
void sgemm(bool trans_a, bool trans_b, int m, int n, int k,
           float alpha, const float* a, int lda, const float* b, int ldb,
           float beta, float* c, int ldc);

/// The library's own thread count, or 0 when it cannot report one.
int  get_threads();
void set_threads(int n);
/// Whether the library exposes thread control. Without it, BLAS cannot be
/// pinned to one thread inside a parallel region, so callers there must not
/// use it.
bool thread_control();

/// Extra directories to search, ahead of the system's default search. The
/// Python module adds the interpreter's prefix and user base, which is where
/// `pip install mkl` puts mkl_rt.
void set_search_dirs(const std::vector<std::string>& dirs);

/// Switch backend now. `which` is "auto", "openblas", "mkl" or "tiled";
/// `path`, if not empty, is the exact library file to load. Returns false and
/// fills `error` if the library cannot be loaded, leaving the current backend
/// unchanged. Not safe to call while another thread is multiplying.
bool select(const std::string& which, const std::string& path, std::string* error);

} // namespace blas
