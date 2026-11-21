// The hot element-wise loops, built with AVX-512 F / BW / DQ / VL (per-file flags). See simd_kernels.inc.
#define SIMD_AVX512
#define SIMD_NS simd_avx512
#include "simd_kernels.inc"
