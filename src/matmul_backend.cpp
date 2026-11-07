#include "ml/matmul_backend.hpp"
#include <algorithm>

#if defined(PROMETHEUS_USE_OPENBLAS)
    #include <cblas.h>
#elif defined(PROMETHEUS_USE_MKL)
    #include <mkl_cblas.h>
#endif

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
    #ifdef PROMETHEUS_USE_OPENMP
    #pragma omp parallel for schedule(dynamic) if(worth_parallel(m, inner, kdim))
    #endif
    for (int i0 = 0; i0 < m; i0 += TILE)
        for (int j0 = 0; j0 < inner; j0 += TILE) {
            int i_end = std::min(i0 + TILE, m);
            int j_end = std::min(j0 + TILE, inner);
            for (int k0 = 0; k0 < kdim; k0 += TILE) {
                int k_end = std::min(k0 + TILE, kdim);
                for (int i = i0; i < i_end; i++)
                    for (int j = j0; j < j_end; j++) {
                        float acc = 0.0f;
                        const float* gi = g + i * kdim;
                        const float* bj = b + j * kdim;
                        for (int kk = k0; kk < k_end; kk++)
                            acc += gi[kk] * bj[kk];
                        out[i * inner + j] += acc;
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

void matmul_forward(int m, int n, int k,
                    const float* a, const float* b, float* c) {
#if defined(PROMETHEUS_USE_OPENBLAS) || defined(PROMETHEUS_USE_MKL)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
#else
    matmul_tiled(m, n, k, a, b, c);
#endif
}

void matmul_backward_a(int m, int inner, int b_cols,
                       const float* grad_out, const float* b_data, float* a_grad) {
#if defined(PROMETHEUS_USE_OPENBLAS) || defined(PROMETHEUS_USE_MKL)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, inner, b_cols,
                1.0f, grad_out, b_cols, b_data, b_cols,
                1.0f, a_grad, inner);
#else
    matmul_bt_tiled(m, inner, b_cols, grad_out, b_data, a_grad);
#endif
}

void matmul_backward_b(int m, int inner, int b_cols,
                       const float* grad_out, const float* a_data, float* b_grad) {
#if defined(PROMETHEUS_USE_OPENBLAS) || defined(PROMETHEUS_USE_MKL)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                inner, b_cols, m,
                1.0f, a_data, inner, grad_out, b_cols,
                1.0f, b_grad, b_cols);
#else
    matmul_at_tiled(m, inner, b_cols, a_data, grad_out, b_grad);
#endif
}
