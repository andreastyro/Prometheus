#pragma once

/// Fast matmul dispatch. The BLAS library -- OpenBLAS, Intel MKL, or none --
/// is chosen at runtime (see ml/blas.hpp); without one, a tiled
/// cache-friendly kernel is used, parallelised across cores when built with
/// PROMETHEUS_USE_OPENMP.

/// Forward pass: C = A @ B
/// A [m, k],  B [k, n],  C [m, n]  — C is zeroed before accumulation
/// Which matmul backend is in use right now: "openblas", "mkl" or "tiled".
/// Worth checking before reading any benchmark: a run that silently fell back
/// to the portable kernel looks identical from the outside, and once produced
/// a phantom 4x regression.
/// While one of these is alive, BLAS runs on the calling thread only, so
/// gemms issued from inside a parallel region use BLAS's kernels rather than
/// falling back to the portable tiled loop. Create it outside the region:
/// changing BLAS's thread count is not itself thread-safe.
///
/// This is how attention gets the effect of a batched gemm: 48 small
/// per-head matmuls, each on one core through a tuned kernel, all at once.
class SerialBlasScope {
public:
    SerialBlasScope();
    ~SerialBlasScope();
    SerialBlasScope(const SerialBlasScope&) = delete;
    SerialBlasScope& operator=(const SerialBlasScope&) = delete;
private:
    int prev_ = 0;
};

/// C = op(A) @ op(B) + beta * C with explicit row strides, on the calling
/// thread (inside a SerialBlasScope it goes through BLAS).
///
/// op(A) is m x k and op(B) is k x n; a transposed operand is stored the
/// other way round. lda/ldb/ldc are the distance between consecutive rows in
/// memory, so a matrix can be a block of columns inside a wider buffer. That
/// is what lets attention read each head straight out of the fused QKV
/// buffer, where a head's row is D floats followed by the other heads, and
/// write each head's output straight into its columns of the concatenation.
void gemm_serial(bool trans_a, bool trans_b, int m, int n, int k,
                 const float* a, int lda, const float* b, int ldb,
                 float beta, float* c, int ldc);

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
