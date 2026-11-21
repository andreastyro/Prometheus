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

} // namespace simd
