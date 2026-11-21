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
/// One LSTM step, a batch row (single-threaded: the caller splits the batch).
/// a: pre-activations i f o g; g: activated gates; c, h: the new cell / output.
void lstm_fwd_row(const float* a, float* g, const float* c_prev, float* c, float* h, int H);
/// da: the gates' gradients; dc: in from step t + 1, out to step t - 1.
void lstm_bwd_row(const float* g, const float* c, const float* c_prev, const float* dh, float* da, float* dc,
                  int H);

} // namespace simd
