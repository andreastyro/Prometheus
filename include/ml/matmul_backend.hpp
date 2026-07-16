#pragma once

/// Fast matmul dispatch — which backend is compiled in depends on build flags:
///   PROMETHEUS_USE_OPENBLAS — cblas_sgemm from OpenBLAS (best on AMD)
///   PROMETHEUS_USE_MKL      — cblas_sgemm from Intel MKL (best on Intel)
///   PROMETHEUS_USE_OPENMP   — tiled matmul parallelised across CPU cores
///   (none)                  — plain tiled cache-friendly matmul (no dependencies)

/// Forward pass: C = A @ B
/// A [m, k],  B [k, n],  C [m, n]  — C is zeroed before accumulation
void matmul_forward(int m, int n, int k,
                    const float* a, const float* b, float* c);

/// Backward pass: accumulate dA += grad_out @ B^T into a_grad
void matmul_backward_a(int m, int inner, int b_cols,
                       const float* grad_out, const float* b_data, float* a_grad);

/// Backward pass: accumulate dB += A^T @ grad_out into b_grad
void matmul_backward_b(int m, int inner, int b_cols,
                       const float* grad_out, const float* a_data, float* b_grad);
