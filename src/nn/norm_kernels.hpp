#pragma once
// Reductions shared by BatchNorm and GroupNorm. Private to src/nn.
//
// A float sum is a reduction the compiler may not reorder without fast-math,
// so a plain loop runs one add at a time, each waiting on the last. These keep
// a vector of partial sums instead (AVX 8 lanes, SSE 4, or 8 scalar lanes the
// compiler can map to NEON) and fold each block of 1024 into a double, so the
// rounding error stays that of a 1024-term float sum however long the row is.
#include <algorithm>
#include <cstdint>

#if defined(__AVX__)
#include <immintrin.h>
#define NORMK_AVX 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define NORMK_SSE 1
#endif

namespace normk {

constexpr int64_t BLOCK = 1024;

#if NORMK_AVX
struct Vec {
    __m256 v;
    static constexpr int W = 8;
    static Vec zero()               { return {_mm256_setzero_ps()}; }
    static Vec set(float a)         { return {_mm256_set1_ps(a)}; }
    static Vec load(const float* p) { return {_mm256_loadu_ps(p)}; }
    friend Vec operator+(Vec a, Vec b) { return {_mm256_add_ps(a.v, b.v)}; }
    friend Vec operator-(Vec a, Vec b) { return {_mm256_sub_ps(a.v, b.v)}; }
    friend Vec operator*(Vec a, Vec b) { return {_mm256_mul_ps(a.v, b.v)}; }
    double hsum() const {
        alignas(32) float t[8];
        _mm256_store_ps(t, v);
        return ((double)t[0] + t[1] + t[2] + t[3]) + ((double)t[4] + t[5] + t[6] + t[7]);
    }
};
#elif NORMK_SSE
struct Vec {
    __m128 v;
    static constexpr int W = 4;
    static Vec zero()               { return {_mm_setzero_ps()}; }
    static Vec set(float a)         { return {_mm_set1_ps(a)}; }
    static Vec load(const float* p) { return {_mm_loadu_ps(p)}; }
    friend Vec operator+(Vec a, Vec b) { return {_mm_add_ps(a.v, b.v)}; }
    friend Vec operator-(Vec a, Vec b) { return {_mm_sub_ps(a.v, b.v)}; }
    friend Vec operator*(Vec a, Vec b) { return {_mm_mul_ps(a.v, b.v)}; }
    double hsum() const {
        alignas(16) float t[4];
        _mm_store_ps(t, v);
        return ((double)t[0] + t[1]) + ((double)t[2] + t[3]);
    }
};
#else
struct Vec {
    float v[8];
    static constexpr int W = 8;
    static Vec zero()               { Vec r; for (int i = 0; i < 8; i++) r.v[i] = 0.0f; return r; }
    static Vec set(float a)         { Vec r; for (int i = 0; i < 8; i++) r.v[i] = a; return r; }
    static Vec load(const float* p) { Vec r; for (int i = 0; i < 8; i++) r.v[i] = p[i]; return r; }
    friend Vec operator+(Vec a, Vec b) { for (int i = 0; i < 8; i++) a.v[i] += b.v[i]; return a; }
    friend Vec operator-(Vec a, Vec b) { for (int i = 0; i < 8; i++) a.v[i] -= b.v[i]; return a; }
    friend Vec operator*(Vec a, Vec b) { for (int i = 0; i < 8; i++) a.v[i] *= b.v[i]; return a; }
    double hsum() const { double s = 0; for (int i = 0; i < 8; i++) s += v[i]; return s; }
};
#endif

/// sum of x[0..n)
inline double sum(const float* x, int64_t n) {
    constexpr int W = Vec::W;
    double total = 0.0;
    for (int64_t b0 = 0; b0 < n; b0 += BLOCK) {
        const int64_t b1 = std::min(n, b0 + BLOCK);
        Vec a0 = Vec::zero(), a1 = Vec::zero();
        int64_t i = b0;
        for (; i + 2 * W <= b1; i += 2 * W) {
            a0 = a0 + Vec::load(x + i);
            a1 = a1 + Vec::load(x + i + W);
        }
        float tail = 0.0f;
        for (; i < b1; i++) tail += x[i];
        total += (a0 + a1).hsum() + tail;
    }
    return total;
}

/// Mean and sum of squared deviations (M2) of x[0..n), two passes. The second
/// pass is taken about the float-rounded mean c and corrected by (sum d)^2 / n,
/// which removes the error of c not being the exact mean.
inline void moments(const float* x, int64_t n, double& mean, double& m2) {
    constexpr int W = Vec::W;
    mean = n ? sum(x, n) / (double)n : 0.0;
    const float c = (float)mean;
    const Vec cv = Vec::set(c);
    double s1 = 0.0, s2 = 0.0;
    for (int64_t b0 = 0; b0 < n; b0 += BLOCK) {
        const int64_t b1 = std::min(n, b0 + BLOCK);
        Vec d0 = Vec::zero(), d1 = Vec::zero(), q0 = Vec::zero(), q1 = Vec::zero();
        int64_t i = b0;
        for (; i + 2 * W <= b1; i += 2 * W) {
            const Vec u = Vec::load(x + i) - cv, w = Vec::load(x + i + W) - cv;
            d0 = d0 + u;     d1 = d1 + w;
            q0 = q0 + u * u; q1 = q1 + w * w;
        }
        float td = 0.0f, tq = 0.0f;
        for (; i < b1; i++) { const float u = x[i] - c; td += u; tq += u * u; }
        s1 += (d0 + d1).hsum() + td;
        s2 += (q0 + q1).hsum() + tq;
    }
    m2 = n ? std::max(0.0, s2 - s1 * s1 / (double)n) : 0.0;
}

/// sum of g[i], and of g[i] * (x[i] - c): what a normalisation backward needs.
inline void grad_sums(const float* g, const float* x, int64_t n, float c,
                      double& sg, double& sgx) {
    constexpr int W = Vec::W;
    const Vec cv = Vec::set(c);
    sg = sgx = 0.0;
    for (int64_t b0 = 0; b0 < n; b0 += BLOCK) {
        const int64_t b1 = std::min(n, b0 + BLOCK);
        Vec a0 = Vec::zero(), a1 = Vec::zero(), p0 = Vec::zero(), p1 = Vec::zero();
        int64_t i = b0;
        for (; i + 2 * W <= b1; i += 2 * W) {
            const Vec g0 = Vec::load(g + i), g1 = Vec::load(g + i + W);
            a0 = a0 + g0; a1 = a1 + g1;
            p0 = p0 + g0 * (Vec::load(x + i) - cv);
            p1 = p1 + g1 * (Vec::load(x + i + W) - cv);
        }
        float ta = 0.0f, tp = 0.0f;
        for (; i < b1; i++) { ta += g[i]; tp += g[i] * (x[i] - c); }
        sg  += (a0 + a1).hsum() + ta;
        sgx += (p0 + p1).hsum() + tp;
    }
}

/// y = a * (x - c) + b. Centred first: a*x + (b - a*c) is one op shorter but
/// cancels when |mean| >> std, and costs the same once memory-bound.
inline void affine(const float* x, float* y, int64_t n, float a, float c, float b) {
    for (int64_t i = 0; i < n; i++) y[i] = a * (x[i] - c) + b;
}

/// gx += k1 * g + k2 * (x - c) + k3
inline void affine_grad(const float* g, const float* x, float* gx, int64_t n,
                        float k1, float k2, float c, float k3) {
    for (int64_t i = 0; i < n; i++) gx[i] += k1 * g[i] + k2 * (x[i] - c) + k3;
}

} // namespace normk
