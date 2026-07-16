#include "ml/matmul_backend.hpp"
#include <algorithm>

#if defined(PROMETHEUS_USE_OPENBLAS)
    #include <cblas.h>
#elif defined(PROMETHEUS_USE_MKL)
    #include <mkl_cblas.h>
#endif

static constexpr int TILE = 64;

// Tiled cache-friendly matmul — always compiled in as fallback
static void matmul_tiled(int m, int n, int k,
                         const float* a, const float* b, float* c) {
    #ifdef PROMETHEUS_USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
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

void matmul_forward(int m, int n, int k,
                    const float* a, const float* b, float* c) {
#if defined(PROMETHEUS_USE_OPENBLAS) || defined(PROMETHEUS_USE_MKL)
    if (m > 128 || n > 128 || k > 128) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
        return;
    }
#endif
    matmul_tiled(m, n, k, a, b, c);
}

void matmul_backward_a(int m, int inner, int b_cols,
                       const float* grad_out, const float* b_data, float* a_grad) {
#if defined(PROMETHEUS_USE_OPENBLAS) || defined(PROMETHEUS_USE_MKL)
    if (m > 128 || inner > 128 || b_cols > 128) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, inner, b_cols,
                    1.0f, grad_out, b_cols, b_data, b_cols,
                    1.0f, a_grad, inner);
        return;
    }
#endif
    for (int i = 0; i < m; i++)
        for (int j = 0; j < inner; j++)
            for (int kk = 0; kk < b_cols; kk++)
                a_grad[i * inner + j] += grad_out[i * b_cols + kk] * b_data[j * b_cols + kk];
}

void matmul_backward_b(int m, int inner, int b_cols,
                       const float* grad_out, const float* a_data, float* b_grad) {
#if defined(PROMETHEUS_USE_OPENBLAS) || defined(PROMETHEUS_USE_MKL)
    if (m > 128 || inner > 128 || b_cols > 128) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    inner, b_cols, m,
                    1.0f, a_data, inner, grad_out, b_cols,
                    1.0f, b_grad, b_cols);
        return;
    }
#endif
    for (int i = 0; i < inner; i++)
        for (int j = 0; j < b_cols; j++)
            for (int kk = 0; kk < m; kk++)
                b_grad[i * b_cols + j] += a_data[kk * inner + i] * grad_out[kk * b_cols + j];
}
