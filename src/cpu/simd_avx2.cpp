// The hot element-wise loops, built with AVX2 + FMA (setup.py / CMakeLists: per-file flags). See simd_kernels.inc.
#define SIMD_AVX2
#define SIMD_NS simd_avx2
#include "simd_kernels.inc"
