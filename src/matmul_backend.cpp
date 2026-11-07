#include "ml/matmul_backend.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <atomic>
#ifdef PROMETHEUS_USE_OPENMP
    #include <omp.h>
#endif

#include "ml/blas.hpp"

static constexpr int TILE = 64;

// Below this many multiply-adds, run serially.
//
// The parallel region itself costs something: a fork and a join per call. A
// transformer step issues tens of small matmuls -- at an embedding width of 384
// each is 384x384, which is only 36 tiles -- and handing 36 tiles to 16 threads
// loses more to synchronisation than it gains. Measured on a 16.8M-parameter
// GPT, throughput peaked at 2 threads and fell back at 4 and above, which is
// the signature of exactly this.
static constexpr long long PARALLEL_MIN_WORK = 2LL * 1024 * 1024;

static inline bool worth_parallel(int m, int n, int k) {
    return (long long)m * (long long)n * (long long)k >= PARALLEL_MIN_WORK;
}

// Tiled cache-friendly matmul — always compiled in as fallback
static void matmul_tiled(int m, int n, int k,
                         const float* a, const float* b, float* c) {
    #ifdef PROMETHEUS_USE_OPENMP
    #pragma omp parallel for schedule(dynamic) if(worth_parallel(m, n, k))
    #endif
    for (int i0 = 0; i0 < m; i0 += TILE)
        for (int k0 = 0; k0 < k; k0 += TILE)
            for (int j0 = 0; j0 < n; j0 += TILE) {
                int i_end = std::min(i0 + TILE, m);
                int k_end = std::min(k0 + TILE, k);
                int j_end = std::min(j0 + TILE, n);
                for (int i = i0; i < i_end; i++)
                    for (int kk = k0; kk < k_end; kk++) {
                        float a_ik = a[i * k + kk];
                        for (int j = j0; j < j_end; j++)
                            c[i * n + j] += a_ik * b[kk * n + j];
                    }
            }
}

// dA += grad_out @ B^T
//   grad_out [m, kdim], b [inner, kdim], out [m, inner]
// Both operands are read along kdim, so keeping kdim innermost walks two rows
// at once. Accumulates into out, so the caller's existing gradient survives.
static void matmul_bt_tiled(int m, int inner, int kdim,
                            const float* g, const float* b, float* out) {
    // out[m, inner] += g[m, kdim] @ b[inner, kdim]^T
    //
    // The obvious loop -- accumulate g[i][kk]*b[j][kk] over kk -- reads both
    // operands contiguously and looks ideal, but it is a floating-point
    // reduction into a single scalar, and the compiler may not reassociate
    // that, so it stays scalar. Measured at 128x128x64 it ran 2.6x slower
    // than transposing b and calling matmul_tiled.
    //
    // So pack a tile of b^T into a contiguous buffer and use the same
    // j-innermost axpy that matmul_tiled uses, which vectorises. The pack is
    // TILE^2 elements and is amortised over the rows of the tile.
    #ifdef PROMETHEUS_USE_OPENMP
    #pragma omp parallel for schedule(dynamic) if(worth_parallel(m, inner, kdim))
    #endif
    for (int i0 = 0; i0 < m; i0 += TILE) {
        float bt[TILE * TILE];
        int i_end = std::min(i0 + TILE, m);
        for (int k0 = 0; k0 < kdim; k0 += TILE)
            for (int j0 = 0; j0 < inner; j0 += TILE) {
                int k_end = std::min(k0 + TILE, kdim);
                int j_end = std::min(j0 + TILE, inner);

                for (int j = j0; j < j_end; j++)
                    for (int kk = k0; kk < k_end; kk++)
                        bt[(kk - k0) * TILE + (j - j0)] = b[j * kdim + kk];

                for (int i = i0; i < i_end; i++)
                    for (int kk = k0; kk < k_end; kk++) {
                        float g_ik = g[i * kdim + kk];
                        const float* btr = bt + (kk - k0) * TILE - j0;
                        for (int j = j0; j < j_end; j++)
                            out[i * inner + j] += g_ik * btr[j];
                    }
            }
    }
}

// dB += A^T @ grad_out
//   a [m, inner], grad_out [m, ncols], out [inner, ncols]
// Parallelised over the output rows so no two threads touch the same element;
// the reduction runs over m inside each tile.
static void matmul_at_tiled(int m, int inner, int ncols,
                            const float* a, const float* g, float* out) {
    #ifdef PROMETHEUS_USE_OPENMP
    #pragma omp parallel for schedule(dynamic) if(worth_parallel(inner, ncols, m))
    #endif
    for (int i0 = 0; i0 < inner; i0 += TILE)
        for (int k0 = 0; k0 < m; k0 += TILE)
            for (int j0 = 0; j0 < ncols; j0 += TILE) {
                int i_end = std::min(i0 + TILE, inner);
                int k_end = std::min(k0 + TILE, m);
                int j_end = std::min(j0 + TILE, ncols);
                for (int i = i0; i < i_end; i++)
                    for (int kk = k0; kk < k_end; kk++) {
                        float a_ki = a[kk * inner + i];
                        const float* gk = g + kk * ncols;
                        float* oi = out + i * ncols;
                        for (int j = j0; j < j_end; j++)
                            oi[j] += a_ki * gk[j];
                    }
            }
}

// A BLAS gemm brings its own thread pool. Called from inside an OpenMP
// parallel region, each of our N threads asks BLAS for N more, and the machine
// stops making progress -- attention's per-head matmuls did exactly this and
// turned a 3.5 s training step into one that never finished.
//
// Rather than rely on every caller remembering, the BLAS path checks for
// itself and falls back to the portable tiled kernel when it is already inside
// a parallel region. The explicit *_serial entry points remain for callers
// that want to state the intent.
static std::atomic<int> g_serial_blas{0};

// BLAS has been pinned to one thread by a live SerialBlasScope, so it is safe
// to call from inside a parallel region.
static inline bool serial_blas_active() {
    return g_serial_blas.load(std::memory_order_relaxed) > 0
        && blas::available() && blas::thread_control();
}

// A BLAS gemm brings its own thread pool. Called from inside an OpenMP
// parallel region, each of our N threads asks BLAS for N more, and the machine
// stops making progress -- attention's per-head matmuls did exactly this and
// turned a 3.5 s training step into one that never finished. So the BLAS path
// checks for itself and uses the portable kernel there, unless a
// SerialBlasScope has pinned BLAS to one thread.
static inline bool nested_call() {
#ifdef PROMETHEUS_USE_OPENMP
    return omp_in_parallel() != 0 && !serial_blas_active();
#else
    return false;
#endif
}

// BLAS for this call: a library is loaded, and calling it here cannot
// oversubscribe the machine.
static inline bool use_blas() {
    return blas::available() && !nested_call();
}

SerialBlasScope::SerialBlasScope() {
    if (g_serial_blas.fetch_add(1) == 0 && blas::available() && blas::thread_control()) {
        prev_ = blas::get_threads();
        blas::set_threads(1);
    }
}

SerialBlasScope::~SerialBlasScope() {
    if (g_serial_blas.fetch_sub(1) == 1 && prev_ > 0)
        blas::set_threads(prev_);
}

void gemm_serial(bool trans_a, bool trans_b, int m, int n, int k,
                 const float* a, int lda, const float* b, int ldb,
                 float beta, float* c, int ldc) {
    if (use_blas()) {
        blas::sgemm(trans_a, trans_b, m, n, k, 1.0f, a, lda, b, ldb, beta, c, ldc);
        return;
    }
    for (int i = 0; i < m; i++) {
        float* ci = c + (size_t)i * ldc;
        if (beta == 0.0f)      for (int j = 0; j < n; j++) ci[j] = 0.0f;
        else if (beta != 1.0f) for (int j = 0; j < n; j++) ci[j] *= beta;
    }

    // Every case is arranged so the innermost loop is an axpy over a row of
    // C, which vectorises; a dot product over k would be a float reduction
    // the compiler will not reorder.
    if (!trans_a && !trans_b) {
        for (int i = 0; i < m; i++) {
            float* ci = c + (size_t)i * ldc;
            for (int kk = 0; kk < k; kk++) {
                const float aik = a[(size_t)i * lda + kk];
                const float* bk = b + (size_t)kk * ldb;
                for (int j = 0; j < n; j++) ci[j] += aik * bk[j];
            }
        }
    } else if (trans_a && !trans_b) {
        for (int kk = 0; kk < k; kk++) {
            const float* ak = a + (size_t)kk * lda;
            const float* bk = b + (size_t)kk * ldb;
            for (int i = 0; i < m; i++) {
                const float aki = ak[i];
                float* ci = c + (size_t)i * ldc;
                for (int j = 0; j < n; j++) ci[j] += aki * bk[j];
            }
        }
    } else if (!trans_a && trans_b) {
        // pack B^T to [k, n] once so the inner loop is the same axpy
        std::vector<float> bt((size_t)k * n);
        for (int j = 0; j < n; j++)
            for (int kk = 0; kk < k; kk++)
                bt[(size_t)kk * n + j] = b[(size_t)j * ldb + kk];
        for (int i = 0; i < m; i++) {
            float* ci = c + (size_t)i * ldc;
            for (int kk = 0; kk < k; kk++) {
                const float aik = a[(size_t)i * lda + kk];
                const float* btk = bt.data() + (size_t)kk * n;
                for (int j = 0; j < n; j++) ci[j] += aik * btk[j];
            }
        }
    } else {
        throw std::runtime_error("gemm_serial: transposing both operands is not supported");
    }
}

const char* matmul_backend_name() {
    return blas::available() ? blas::name().c_str() : "tiled";
}

int matmul_max_threads() {
#ifdef PROMETHEUS_USE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void matmul_forward(int m, int n, int k,
                    const float* a, const float* b, float* c) {
    if (use_blas()) {
        blas::sgemm(false, false, m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
        return;
    }
    // the tiled kernel accumulates, so match the overwrite contract
    for (int i = 0; i < m * n; i++) c[i] = 0.0f;
    matmul_tiled(m, n, k, a, b, c);
}

void matmul_forward_bt(int m, int n, int k,
                       const float* a, const float* b, float* c) {
    if (use_blas()) {
        // the transpose is a flag, not a copy
        blas::sgemm(false, true, m, n, k, 1.0f, a, k, b, k, 0.0f, c, n);
        return;
    }
    for (int i = 0; i < m * n; i++) c[i] = 0.0f;
    matmul_bt_tiled(m, n, k, a, b, c);
}

void matmul_accumulate(int m, int n, int k,
                       const float* a, const float* b, float* c) {
    if (use_blas()) {
        blas::sgemm(false, false, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
        return;
    }
    matmul_tiled(m, n, k, a, b, c);   // already accumulates
}

void matmul_backward_a(int m, int inner, int b_cols,
                       const float* grad_out, const float* b_data, float* a_grad) {
    if (use_blas()) {
        blas::sgemm(false, true, m, inner, b_cols,
                    1.0f, grad_out, b_cols, b_data, b_cols, 1.0f, a_grad, inner);
        return;
    }
    matmul_bt_tiled(m, inner, b_cols, grad_out, b_data, a_grad);
}

void matmul_backward_b(int m, int inner, int b_cols,
                       const float* grad_out, const float* a_data, float* b_grad) {
    if (use_blas()) {
        blas::sgemm(true, false, inner, b_cols, m,
                    1.0f, a_data, inner, grad_out, b_cols, 1.0f, b_grad, b_cols);
        return;
    }
    matmul_at_tiled(m, inner, b_cols, a_data, grad_out, b_grad);
}
