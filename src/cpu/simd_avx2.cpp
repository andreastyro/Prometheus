// The hot element-wise loops, built with AVX2 + FMA (setup.py / CMakeLists: per-file flags). See simd_kernels.inc.
// x86 only: elsewhere (ARM) this file is the portable code again, never chosen
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define SIMD_AVX2
#endif
#define SIMD_NS simd_avx2
#include "simd_kernels.inc"
