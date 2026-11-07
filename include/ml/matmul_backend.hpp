#pragma once

/// Fast matmul dispatch — which backend is compiled in depends on build flags:
///   PROMETHEUS_USE_OPENBLAS — cblas_sgemm from OpenBLAS (best on AMD)
///   PROMETHEUS_USE_MKL      — cblas_sgemm from Intel MKL (best on Intel)
///   PROMETHEUS_USE_OPENMP   — tiled matmul parallelised across CPU cores
///   (none)                  — plain tiled cache-friendly matmul (no dependencies)

/// Forward pass: C = A @ B
/// A [m, k],  B [k, n],  C [m, n]  — C is zeroed before accumulation
/// Which matmul backend this build actually compiled in, and whether it can
/// use more than one core. Worth having: the backend is chosen by compile-time
/// macros with every option defaulting to off, so a build that silently fell
/// back to the portable kernel looks identical from the outside. Reading a
/// benchmark without checking this produced a phantom 4x regression once.
/// Serial variants, for small matrices multiplied from inside a parallel
/// region. A BLAS gemm brings its own thread pool, so calling one from each
/// of 16 OpenMP threads asks for 256 threads and the machine stops making
/// progress -- attention's per-head matmuls hit exactly this. These always
/// use the portable tiled kernel and spawn nothing.
void matmul_forward_serial(int m, int n, int k,
                           const float* a, const float* b, float* c);
void matmul_forward_bt_serial(int m, int n, int k,
                              const float* a, const float* b, float* c);
void matmul_backward_a_serial(int m, int inner, int b_cols,
                              const float* grad_out, const float* b_data, float* a_grad);
void matmul_backward_b_serial(int m, int inner, int b_cols,
                              const float* grad_out, const float* a_data, float* b_grad);

const char* matmul_backend_name();

/// Threads OpenMP will actually use, or 1 when built without it.
int matmul_max_threads();

void matmul_forward(int m, int n, int k,
                    const float* a, const float* b, float* c);

/// Forward with the right operand transposed: C = A @ B^T
///   A [m, k],  B [n, k],  C [m, n]  — C is overwritten
///
/// This is what PyTorch's nn.Linear does: it stores the weight as
/// [out, in] and hands a transpose flag to gemm rather than materialising the
/// transpose. BLAS reads the matrix transposed while striding over the
/// original memory, so the transpose costs nothing.
///
/// Weight tying needs exactly this: logits = x @ tok_emb.weight^T. Building
/// the transpose explicitly copied 24.6 MB per forward pass on a 16000x384
/// embedding table, about 10% of the output projection.
void matmul_forward_bt(int m, int n, int k,
                       const float* a, const float* b, float* c);

/// C += A @ B, accumulating rather than overwriting.
///   A [m, k],  B [k, n],  C [m, n]
void matmul_accumulate(int m, int n, int k,
                       const float* a, const float* b, float* c);

/// Backward pass: accumulate dA += grad_out @ B^T into a_grad
void matmul_backward_a(int m, int inner, int b_cols,
                       const float* grad_out, const float* b_data, float* a_grad);

/// Backward pass: accumulate dB += A^T @ grad_out into b_grad
void matmul_backward_b(int m, int inner, int b_cols,
                       const float* grad_out, const float* a_data, float* b_grad);
