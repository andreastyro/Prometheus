#pragma once
// Hot element-wise CPU loops, built three ways (AVX-512, AVX2 + FMA, portable)
// and chosen once at startup -- AVX2 where the CPU has it; each call is
// split across the OpenMP threads. PROMETHEUS_SIMD=base|avx2|avx512 forces one
// (an unsupported choice falls back). Internal.

namespace simd {

/// "avx512", "avx2" or "base": what the loops below run as.
const char* level();
void gelu_fwd(const float* x, float* y, long long n);
/// gx += gy * gelu'(x)
void gelu_bwd(const float* x, const float* gy, float* gx, long long n);
/// One LSTM step for n units of a batch row (single-threaded), the four gates
/// H apart. a: pre-activations i f o g (may be g itself); g: activated gates;
/// c, h: the new cell / output.
void lstm_fwd_row(const float* a, float* g, const float* c_prev, float* c, float* h, int n, int H);
/// da: the gates' gradients; dc: in from step t + 1, out to step t - 1.
void lstm_bwd_row(const float* g, const float* c, const float* c_prev, const float* dh, float* da, float* dc,
                  int n, int H);

/// float32 <-> bfloat16 / float16, contiguous (bfloat16: round to nearest
/// even, as the scalar path). The float16 ones need F16C: false when only the
/// portable build is available, and nothing was converted.
void f32_to_bf16(const float* x, unsigned short* y, long long n);
void bf16_to_f32(const unsigned short* x, float* y, long long n);
bool f32_to_f16(const float* x, unsigned short* y, long long n);
bool f16_to_f32(const unsigned short* x, float* y, long long n);

/// C[m, n] += op(A) op(B), row-major (ta: A stored [k, m]; tb: B stored
/// [n, k]) -- the matmul used when no BLAS library is loaded: packed panels
/// and a register-blocked AVX2 / FMA micro-kernel. threads: split the rows
/// over OpenMP (ignored inside a parallel region).
void gemm_acc(bool ta, bool tb, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c,
              int ldc, bool threads = true);

} // namespace simd
