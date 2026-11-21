// Runtime choice between the three builds of simd_kernels.inc.
#include "simd.hpp"
#include <cstdlib>
#include <cstring>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace simd_base {
void gelu_fwd(const float*, float*, long long);
void gelu_bwd(const float*, const float*, float*, long long);
void lstm_fwd_row(const float*, float*, const float*, float*, float*, int);
void lstm_bwd_row(const float*, const float*, const float*, const float*, float*, float*, int);
}
namespace simd_avx2 {
void gelu_fwd(const float*, float*, long long);
void gelu_bwd(const float*, const float*, float*, long long);
void lstm_fwd_row(const float*, float*, const float*, float*, float*, int);
void lstm_bwd_row(const float*, const float*, const float*, const float*, float*, float*, int);
}
namespace simd_avx512 {
void gelu_fwd(const float*, float*, long long);
void gelu_bwd(const float*, const float*, float*, long long);
void lstm_fwd_row(const float*, float*, const float*, float*, float*, int);
void lstm_bwd_row(const float*, const float*, const float*, const float*, float*, float*, int);
}

namespace {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
void cpuid(int leaf, int sub, unsigned r[4]) {
#ifdef _MSC_VER
    int x[4];
    __cpuidex(x, leaf, sub);
    for (int i = 0; i < 4; i++) r[i] = (unsigned)x[i];
#else
    unsigned a, b, c, d;
    __asm__ __volatile__("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(leaf), "c"(sub));
    r[0] = a; r[1] = b; r[2] = c; r[3] = d;
#endif
}
unsigned long long xgetbv0() {
#ifdef _MSC_VER
    return _xgetbv(0);
#else
    unsigned lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
#endif
}
// 2: AVX-512 F/BW/DQ/VL usable, 1: AVX2 + FMA, 0: neither. The OS must save
// the wide registers (XCR0), not only the CPU have the instructions.
int detect() {
    unsigned r[4];
    cpuid(0, 0, r);
    if (r[0] < 7) return 0;
    cpuid(1, 0, r);
    const bool osxsave = (r[2] >> 27) & 1, fma = (r[2] >> 12) & 1, avx = (r[2] >> 28) & 1;
    if (!osxsave || !avx) return 0;
    const unsigned long long xcr0 = xgetbv0();
    if ((xcr0 & 6) != 6) return 0;
    cpuid(7, 0, r);
    const bool avx2 = (r[1] >> 5) & 1;
    const bool f = (r[1] >> 16) & 1, dq = (r[1] >> 17) & 1, bw = (r[1] >> 30) & 1, vl = (r[1] >> 31) & 1;
    if (f && dq && bw && vl && (xcr0 & 0xe0) == 0xe0) return 2;
    return avx2 && fma ? 1 : 0;
}
#else
int detect() { return 0; }
#endif

// AVX2 by default even where AVX-512 is there: these loops are memory-bound,
// and on a Zen 5 laptop the AVX-512 build was slower (GELU over 1M: 0.60 vs
// 0.47 ms). PROMETHEUS_SIMD=avx512 asks for it.
int choose() {
    const int has = detect();
    int want = 1;
    if (const char* e = std::getenv("PROMETHEUS_SIMD"))
        want = !std::strcmp(e, "avx512") ? 2 : !std::strcmp(e, "avx2") ? 1 : 0;
    return want < has ? want : has;
}

const int g_level = choose();

// split a loop over the OpenMP threads, each thread one contiguous chunk
template <class F>
void split(long long n, F f) {
    if (n < 32768) { f(0, n); return; }
    #pragma omp parallel
    {
        long long t = 0, nt = 1;
#ifdef _OPENMP
        t = omp_get_thread_num();
        nt = omp_get_num_threads();
#endif
        const long long chunk = (n + nt - 1) / nt, a = t * chunk, b = a + chunk < n ? a + chunk : n;
        if (a < b) f(a, b);
    }
}

} // namespace

namespace simd {

const char* level() { return g_level == 2 ? "avx512" : g_level == 1 ? "avx2" : "base"; }

void gelu_fwd(const float* x, float* y, long long n) {
    split(n, [&](long long a, long long b) {
        if (g_level == 2) simd_avx512::gelu_fwd(x + a, y + a, b - a);
        else if (g_level == 1) simd_avx2::gelu_fwd(x + a, y + a, b - a);
        else simd_base::gelu_fwd(x + a, y + a, b - a);
    });
}

void gelu_bwd(const float* x, const float* gy, float* gx, long long n) {
    split(n, [&](long long a, long long b) {
        if (g_level == 2) simd_avx512::gelu_bwd(x + a, gy + a, gx + a, b - a);
        else if (g_level == 1) simd_avx2::gelu_bwd(x + a, gy + a, gx + a, b - a);
        else simd_base::gelu_bwd(x + a, gy + a, gx + a, b - a);
    });
}

void lstm_fwd_row(const float* a, float* g, const float* c_prev, float* c, float* h, int H) {
    if (g_level == 2) simd_avx512::lstm_fwd_row(a, g, c_prev, c, h, H);
    else if (g_level == 1) simd_avx2::lstm_fwd_row(a, g, c_prev, c, h, H);
    else simd_base::lstm_fwd_row(a, g, c_prev, c, h, H);
}

void lstm_bwd_row(const float* g, const float* c, const float* c_prev, const float* dh, float* da, float* dc,
                  int H) {
    if (g_level == 2) simd_avx512::lstm_bwd_row(g, c, c_prev, dh, da, dc, H);
    else if (g_level == 1) simd_avx2::lstm_bwd_row(g, c, c_prev, dh, da, dc, H);
    else simd_base::lstm_bwd_row(g, c, c_prev, dh, da, dc, H);
}

} // namespace simd
