// Hot element-wise loops, compiled three times -- portable, AVX2 + FMA and
// AVX-512 (simd_base.cpp, simd_avx2.cpp, simd_avx512.cpp, each defining
// SIMD_NS) -- and picked at runtime by simd.cpp from what the CPU supports.
//
// Everything here has internal linkage or lives in SIMD_NS: an inline
// function shared with the rest of the library (std::min, fastmath::exp)
// could be emitted from this file with AVX-512 instructions and chosen by the
// linker for every caller, which would crash an older CPU. So no C++ headers
// (string.h's memcpy is an intrinsic), no templates, plain arithmetic.

#include <string.h>
#if defined(SIMD_AVX512) || defined(SIMD_AVX2)
#include <immintrin.h>
#endif

namespace SIMD_NS {

static inline float fexp(float x) {
    x = x > 88.0f ? 88.0f : (x < -87.0f ? -87.0f : x);
    const float shifter = 12582912.0f;                  // round(x / ln2) by 1.5 * 2^23
    const float n = (x * 1.44269504088896341f + shifter) - shifter;
    float r = x - n * 0.693359375f;
    r = r + n * 2.12194440e-4f;
    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    const int bits = ((int)n + 127) << 23;              // 2^n into the exponent bits
    float scale;
    memcpy(&scale, &bits, sizeof scale);                // an intrinsic: vectorises (a union did not, MSVC)
    return p * scale;
}

static inline float ftanh(float x) { return 1.0f - 2.0f / (fexp(2.0f * x) + 1.0f); }

// Explicit vectors where the build has them: MSVC's auto-vectoriser leaves
// this exp / tanh loop scalar (13 ns an element on one thread). SIMD_WIDTH
// lanes at a time, the tail through the scalar code.
#if defined(SIMD_AVX512)
#define SIMD_WIDTH 16
typedef __m512 vf;
static inline vf vset(float a) { return _mm512_set1_ps(a); }
static inline vf vload(const float* p) { return _mm512_loadu_ps(p); }
static inline void vstore(float* p, vf a) { _mm512_storeu_ps(p, a); }
static inline vf vadd(vf a, vf b) { return _mm512_add_ps(a, b); }
static inline vf vsub(vf a, vf b) { return _mm512_sub_ps(a, b); }
static inline vf vmul(vf a, vf b) { return _mm512_mul_ps(a, b); }
static inline vf vdiv(vf a, vf b) { return _mm512_div_ps(a, b); }
static inline vf vfma(vf a, vf b, vf c) { return _mm512_fmadd_ps(a, b, c); }
static inline vf vclamp(vf a, float lo, float hi) { return _mm512_min_ps(_mm512_max_ps(a, vset(lo)), vset(hi)); }
static inline vf vround(vf a) { return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
static inline vf vpow2(vf n) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23));
}
#elif defined(SIMD_AVX2)
#define SIMD_WIDTH 8
typedef __m256 vf;
static inline vf vset(float a) { return _mm256_set1_ps(a); }
static inline vf vload(const float* p) { return _mm256_loadu_ps(p); }
static inline void vstore(float* p, vf a) { _mm256_storeu_ps(p, a); }
static inline vf vadd(vf a, vf b) { return _mm256_add_ps(a, b); }
static inline vf vsub(vf a, vf b) { return _mm256_sub_ps(a, b); }
static inline vf vmul(vf a, vf b) { return _mm256_mul_ps(a, b); }
static inline vf vdiv(vf a, vf b) { return _mm256_div_ps(a, b); }
static inline vf vfma(vf a, vf b, vf c) { return _mm256_fmadd_ps(a, b, c); }
static inline vf vclamp(vf a, float lo, float hi) { return _mm256_min_ps(_mm256_max_ps(a, vset(lo)), vset(hi)); }
static inline vf vround(vf a) { return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
static inline vf vpow2(vf n) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23));
}
#endif

#ifdef SIMD_WIDTH
// fexp, lane-wise: the same range reduction and polynomial
static inline vf vexp(vf x) {
    x = vclamp(x, -87.0f, 88.0f);
    const vf n = vround(vmul(x, vset(1.44269504088896341f)));
    vf r = vfma(n, vset(-0.693359375f), x);
    r = vfma(n, vset(2.12194440e-4f), r);
    vf p = vset(1.0f / 720.0f);
    p = vfma(p, r, vset(1.0f / 120.0f));
    p = vfma(p, r, vset(1.0f / 24.0f));
    p = vfma(p, r, vset(1.0f / 6.0f));
    p = vfma(p, r, vset(0.5f));
    p = vfma(p, r, vset(1.0f));
    p = vfma(p, r, vset(1.0f));
    return vmul(p, vpow2(n));
}
static inline vf vtanh(vf x) {
    return vsub(vset(1.0f), vdiv(vset(2.0f), vadd(vexp(vmul(vset(2.0f), x)), vset(1.0f))));
}
#endif

void gelu_fwd(const float* x, float* y, long long n) {
    const float k = 0.7978845608f, c = 0.044715f;
    long long i = 0;
#ifdef SIMD_WIDTH
    for (; i + SIMD_WIDTH <= n; i += SIMD_WIDTH) {
        const vf v = vload(x + i);
        const vf inner = vmul(vset(k), vfma(vmul(vset(c), vmul(v, v)), v, v));
        vstore(y + i, vmul(vmul(vset(0.5f), v), vadd(vset(1.0f), vtanh(inner))));
    }
#endif
    for (; i < n; i++) {
        const float v = x[i];
        y[i] = 0.5f * v * (1.0f + ftanh(k * (v + c * v * v * v)));
    }
}

// gx += gy * gelu'(x)
void gelu_bwd(const float* x, const float* gy, float* gx, long long n) {
    const float k = 0.7978845608f, c = 0.044715f;
    long long i = 0;
#ifdef SIMD_WIDTH
    for (; i + SIMD_WIDTH <= n; i += SIMD_WIDTH) {
        const vf v = vload(x + i), v2 = vmul(v, v);
        const vf t = vtanh(vmul(vset(k), vfma(vmul(vset(c), v2), v, v)));
        const vf left = vmul(vset(0.5f), vadd(vset(1.0f), t));
        const vf dinner = vmul(vset(k), vfma(vset(3.0f * c), v2, vset(1.0f)));
        const vf right = vmul(vmul(vmul(vset(0.5f), v), vsub(vset(1.0f), vmul(t, t))), dinner);
        vstore(gx + i, vfma(vload(gy + i), vadd(left, right), vload(gx + i)));
    }
#endif
    for (; i < n; i++) {
        const float v = x[i];
        const float t = ftanh(k * (v + c * v * v * v));
        const float d = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * k * (1.0f + 3.0f * c * v * v);
        gx[i] += gy[i] * d;
    }
}

} // namespace SIMD_NS
