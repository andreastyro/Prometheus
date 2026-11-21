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


// One LSTM step's gates for n units of a batch row, the gates H apart: a = the
// pre-activations (i f o g) in; g = the activated gates, c = the new cell, h = the output out.
static inline float fsig(float x) { return 1.0f / (1.0f + fexp(-x)); }
#ifdef SIMD_WIDTH
static inline vf vsig(vf x) { return vdiv(vset(1.0f), vadd(vset(1.0f), vexp(vmul(vset(-1.0f), x)))); }
#endif
void lstm_fwd_row(const float* a, float* g, const float* c_prev, float* c, float* h, int n, int H) {
    int j = 0;
#ifdef SIMD_WIDTH
    for (; j + SIMD_WIDTH <= n; j += SIMD_WIDTH) {
        const vf i_ = vsig(vload(a + j)), f_ = vsig(vload(a + H + j));
        const vf o_ = vsig(vload(a + 2 * H + j)), g_ = vtanh(vload(a + 3 * H + j));
        vstore(g + j, i_); vstore(g + H + j, f_); vstore(g + 2 * H + j, o_); vstore(g + 3 * H + j, g_);
        const vf ct = vfma(f_, vload(c_prev + j), vmul(i_, g_));
        vstore(c + j, ct);
        vstore(h + j, vmul(o_, vtanh(ct)));
    }
#endif
    for (; j < n; j++) {
        const float i_ = fsig(a[j]), f_ = fsig(a[H + j]), o_ = fsig(a[2 * H + j]), g_ = ftanh(a[3 * H + j]);
        g[j] = i_; g[H + j] = f_; g[2 * H + j] = o_; g[3 * H + j] = g_;
        const float ct = f_ * c_prev[j] + i_ * g_;
        c[j] = ct;
        h[j] = o_ * ftanh(ct);
    }
}

// Its backward for a batch row: g the saved gates, c / c_prev the cells, dh the
// gradient reaching h; da the gates' gradients out, dc in (from t + 1) and out.
void lstm_bwd_row(const float* g, const float* c, const float* c_prev, const float* dh, float* da, float* dc,
                  int n, int H) {
    int j = 0;
#ifdef SIMD_WIDTH
    const vf one = vset(1.0f);
    for (; j + SIMD_WIDTH <= n; j += SIMD_WIDTH) {
        const vf i_ = vload(g + j), f_ = vload(g + H + j), o_ = vload(g + 2 * H + j), g_ = vload(g + 3 * H + j);
        const vf tc = vtanh(vload(c + j)), d = vload(dh + j);
        const vf dct = vfma(vmul(d, o_), vsub(one, vmul(tc, tc)), vload(dc + j));
        vstore(da + j, vmul(vmul(dct, g_), vmul(i_, vsub(one, i_))));
        vstore(da + H + j, vmul(vmul(dct, vload(c_prev + j)), vmul(f_, vsub(one, f_))));
        vstore(da + 2 * H + j, vmul(vmul(d, tc), vmul(o_, vsub(one, o_))));
        vstore(da + 3 * H + j, vmul(vmul(dct, i_), vsub(one, vmul(g_, g_))));
        vstore(dc + j, vmul(dct, f_));
    }
#endif
    for (; j < n; j++) {
        const float i_ = g[j], f_ = g[H + j], o_ = g[2 * H + j], g_ = g[3 * H + j];
        const float tc = ftanh(c[j]);
        const float dct = dc[j] + dh[j] * o_ * (1.0f - tc * tc);
        da[j] = dct * g_ * i_ * (1.0f - i_);
        da[H + j] = dct * c_prev[j] * f_ * (1.0f - f_);
        da[2 * H + j] = dh[j] * tc * o_ * (1.0f - o_);
        da[3 * H + j] = dct * i_ * (1.0f - g_ * g_);
        dc[j] = dct * f_;
    }
}


// float32 <-> bfloat16 / float16, contiguous. bf16: round to nearest even
// (NaN kept quiet), the bit pattern the scalar store uses; f16 through F16C.
static inline unsigned short f2bf_s(float f) {
    unsigned int u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
static inline float bf2f_s(unsigned short h) {
    const unsigned int u = (unsigned int)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
void f32_to_bf16(const float* x, unsigned short* y, long long n) {
    long long i = 0;
#if defined(SIMD_AVX2) || defined(SIMD_AVX512)
    const __m256i bias = _mm256_set1_epi32(0x7fff), one = _mm256_set1_epi32(1);
    const __m256i absmask = _mm256_set1_epi32(0x7fffffff), inf = _mm256_set1_epi32(0x7f800000);
    for (; i + 8 <= n; i += 8) {
        const __m256i u = _mm256_castps_si256(_mm256_loadu_ps(x + i));
        const __m256i rounded = _mm256_add_epi32(u, _mm256_add_epi32(bias, _mm256_and_si256(_mm256_srli_epi32(u, 16), one)));
        const __m256i nan = _mm256_cmpgt_epi32(_mm256_and_si256(u, absmask), inf);
        const __m256i quiet = _mm256_or_si256(u, _mm256_set1_epi32(0x400000));
        const __m256i r = _mm256_srli_epi32(_mm256_blendv_epi8(rounded, quiet, nan), 16);
        // eight 32-bit lanes -> eight 16-bit values
        const __m128i lo = _mm256_castsi256_si128(r), hi = _mm256_extracti128_si256(r, 1);
        _mm_storeu_si128((__m128i*)(y + i), _mm_packus_epi32(lo, hi));
    }
#endif
    for (; i < n; i++) y[i] = f2bf_s(x[i]);
}
void bf16_to_f32(const unsigned short* x, float* y, long long n) {
    long long i = 0;
#if defined(SIMD_AVX2) || defined(SIMD_AVX512)
    for (; i + 8 <= n; i += 8) {
        const __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(x + i)));
        _mm256_storeu_ps(y + i, _mm256_castsi256_ps(_mm256_slli_epi32(w, 16)));
    }
#endif
    for (; i < n; i++) y[i] = bf2f_s(x[i]);
}
// float32 -> float16 in software, exact round to nearest even, subnormals
// included: the F16C instruction follows the flush-to-zero mode some
// libraries in the process switch on, and turned values under 6.1e-5 into 0
static inline unsigned short f2h_s(float x) {
    unsigned int u;
    memcpy(&u, &x, 4);
    const unsigned int s = (u >> 16) & 0x8000u;
    int e = (int)((u >> 23) & 0xff) - 127 + 15;
    unsigned int m = u & 0x7fffff;
    if (((u >> 23) & 0xff) == 0xff) return (unsigned short)(s | 0x7c00u | (m ? 0x200u : 0u));
    if (e >= 31) return (unsigned short)(s | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (unsigned short)s;
        m |= 0x800000u;
        const unsigned int shift = (unsigned int)(14 - e), r0 = m >> shift, rem = m & ((1u << shift) - 1),
                           half = 1u << (shift - 1);
        unsigned int r = r0;
        if (rem > half || (rem == half && (r & 1u))) r++;
        return (unsigned short)(s | r);
    }
    unsigned int r = ((unsigned int)e << 10) | (m >> 13), rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1u))) r++;
    return (unsigned short)(s | r);
}
// float16 needs F16C (every AVX2 CPU has it); the portable build returns
// false and the caller keeps its scalar conversion
bool f32_to_f16(const float* x, unsigned short* y, long long n) {
#if defined(SIMD_AVX2) || defined(SIMD_AVX512)
    long long i = 0;
    const __m256 tiny = _mm256_set1_ps(6.103515625e-05f), absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        const __m256 a = _mm256_and_ps(v, absmask);
        // lanes headed for a half subnormal: in software (see f2h_s)
        const int sub = _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(a, tiny, _CMP_LT_OQ),
                                                         _mm256_cmp_ps(a, _mm256_setzero_ps(), _CMP_GT_OQ)));
        if (sub) { for (int l = 0; l < 8; l++) y[i + l] = f2h_s(x[i + l]); continue; }
        _mm_storeu_si128((__m128i*)(y + i), _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
    }
    for (; i < n; i++) y[i] = f2h_s(x[i]);
    return true;
#else
    (void)x; (void)y; (void)n;
    return false;
#endif
}
bool f16_to_f32(const unsigned short* x, float* y, long long n) {
#if defined(SIMD_AVX2) || defined(SIMD_AVX512)
    long long i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + i))));
    for (; i < n; i++) y[i] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[i])));
    return true;
#else
    (void)x; (void)y; (void)n;
    return false;
#endif
}

// ── matrix multiply without BLAS ──────────────────────────────────────────────
//
// C[m, n] += op(A)[m, k] op(B)[k, n], row-major; ta: A is stored [k, m], tb: B
// is stored [n, k]. The BLAS scheme: B packed in panels of GN columns and A in
// panels of GM rows, KC deep, so the micro-kernel streams both contiguously
// and keeps a GM x GN block of C in registers (AVX2: 12 accumulators of 8).
// work: GMC * GKC + GKC * GNC floats.
#define GM 6
#define GN 16
#define GKC 256
#define GMC 72
#define GNC 1024

static void pack_a(int ta, const float* a, int lda, int i0, int mc, int p0, int kc, float* pa) {
    for (int ip = 0; ip < mc; ip += GM) {
        float* d = pa + (long long)ip * kc;
        for (int p = 0; p < kc; p++)
            for (int r = 0; r < GM; r++) {
                const int i = i0 + ip + r;
                d[p * GM + r] = ip + r < mc ? (ta ? a[(long long)(p0 + p) * lda + i] : a[(long long)i * lda + p0 + p])
                                            : 0.0f;
            }
    }
}

static void pack_b(int tb, const float* b, int ldb, int p0, int kc, int j0, int nc, float* pb) {
    for (int jp = 0; jp < nc; jp += GN) {
        float* d = pb + (long long)jp * kc;
        for (int p = 0; p < kc; p++) {
            const int pp = p0 + p;
            if (!tb && jp + GN <= nc) {
                const float* s = b + (long long)pp * ldb + j0 + jp;
                for (int q = 0; q < GN; q++) d[p * GN + q] = s[q];
            } else {
                for (int q = 0; q < GN; q++) {
                    const int j = j0 + jp + q;
                    d[p * GN + q] = jp + q < nc ? (tb ? b[(long long)j * ldb + pp] : b[(long long)pp * ldb + j]) : 0.0f;
                }
            }
        }
    }
}

// C[mr, nr] (row stride ldc) += pa (kc x GM) . pb (kc x GN)
static void micro(int kc, const float* pa, const float* pb, float* c, int ldc, int mr, int nr) {
#ifdef SIMD_WIDTH
    if (mr == GM && nr == GN) {
        __m256 c00 = _mm256_loadu_ps(c), c01 = _mm256_loadu_ps(c + 8);
        __m256 c10 = _mm256_loadu_ps(c + ldc), c11 = _mm256_loadu_ps(c + ldc + 8);
        __m256 c20 = _mm256_loadu_ps(c + 2 * ldc), c21 = _mm256_loadu_ps(c + 2 * ldc + 8);
        __m256 c30 = _mm256_loadu_ps(c + 3 * ldc), c31 = _mm256_loadu_ps(c + 3 * ldc + 8);
        __m256 c40 = _mm256_loadu_ps(c + 4 * ldc), c41 = _mm256_loadu_ps(c + 4 * ldc + 8);
        __m256 c50 = _mm256_loadu_ps(c + 5 * ldc), c51 = _mm256_loadu_ps(c + 5 * ldc + 8);
        for (int p = 0; p < kc; p++) {
            const __m256 b0 = _mm256_loadu_ps(pb + p * GN), b1 = _mm256_loadu_ps(pb + p * GN + 8);
            const float* ap = pa + p * GM;
            __m256 a;
            a = _mm256_broadcast_ss(ap + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
            a = _mm256_broadcast_ss(ap + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
            a = _mm256_broadcast_ss(ap + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
            a = _mm256_broadcast_ss(ap + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
            a = _mm256_broadcast_ss(ap + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
            a = _mm256_broadcast_ss(ap + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
        }
        _mm256_storeu_ps(c, c00); _mm256_storeu_ps(c + 8, c01);
        _mm256_storeu_ps(c + ldc, c10); _mm256_storeu_ps(c + ldc + 8, c11);
        _mm256_storeu_ps(c + 2 * ldc, c20); _mm256_storeu_ps(c + 2 * ldc + 8, c21);
        _mm256_storeu_ps(c + 3 * ldc, c30); _mm256_storeu_ps(c + 3 * ldc + 8, c31);
        _mm256_storeu_ps(c + 4 * ldc, c40); _mm256_storeu_ps(c + 4 * ldc + 8, c41);
        _mm256_storeu_ps(c + 5 * ldc, c50); _mm256_storeu_ps(c + 5 * ldc + 8, c51);
        return;
    }
#endif
    float acc[GM][GN];
    for (int r = 0; r < GM; r++)
        for (int q = 0; q < GN; q++) acc[r][q] = 0.0f;
    for (int p = 0; p < kc; p++)
        for (int r = 0; r < GM; r++) {
            const float av = pa[p * GM + r];
            for (int q = 0; q < GN; q++) acc[r][q] += av * pb[p * GN + q];
        }
    for (int r = 0; r < mr; r++)
        for (int q = 0; q < nr; q++) c[(long long)r * ldc + q] += acc[r][q];
}

void gemm_acc(int ta, int tb, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c,
              int ldc, float* work) {
    float* pa = work;
    float* pb = work + GMC * GKC;
    for (int j0 = 0; j0 < n; j0 += GNC) {
        const int nc = n - j0 < GNC ? n - j0 : GNC;
        for (int p0 = 0; p0 < k; p0 += GKC) {
            const int kc = k - p0 < GKC ? k - p0 : GKC;
            pack_b(tb, b, ldb, p0, kc, j0, nc, pb);
            for (int i0 = 0; i0 < m; i0 += GMC) {
                const int mc = m - i0 < GMC ? m - i0 : GMC;
                pack_a(ta, a, lda, i0, mc, p0, kc, pa);
                for (int jp = 0; jp < nc; jp += GN)
                    for (int ip = 0; ip < mc; ip += GM)
                        micro(kc, pa + (long long)ip * kc, pb + (long long)jp * kc,
                              c + (long long)(i0 + ip) * ldc + j0 + jp, ldc, mc - ip < GM ? mc - ip : GM,
                              nc - jp < GN ? nc - jp : GN);
            }
        }
    }
}

} // namespace SIMD_NS
