// Runtime choice between the three builds of simd_kernels.inc.
#include "simd.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace simd_base {
void gemm_acc(int, int, int, int, int, const float*, int, const float*, int, float*, int, float*);
void f32_to_bf16(const float*, unsigned short*, long long);
void bf16_to_f32(const unsigned short*, float*, long long);
bool f32_to_f16(const float*, unsigned short*, long long);
bool f16_to_f32(const unsigned short*, float*, long long);
void gelu_fwd(const float*, float*, long long);
void gelu_bwd(const float*, const float*, float*, long long);
void lstm_fwd_row(const float*, float*, const float*, float*, float*, int, int);
void lstm_bwd_row(const float*, const float*, const float*, const float*, float*, float*, int, int);
}
namespace simd_avx2 {
void gemm_acc(int, int, int, int, int, const float*, int, const float*, int, float*, int, float*);
void f32_to_bf16(const float*, unsigned short*, long long);
void bf16_to_f32(const unsigned short*, float*, long long);
bool f32_to_f16(const float*, unsigned short*, long long);
bool f16_to_f32(const unsigned short*, float*, long long);
void gelu_fwd(const float*, float*, long long);
void gelu_bwd(const float*, const float*, float*, long long);
void lstm_fwd_row(const float*, float*, const float*, float*, float*, int, int);
void lstm_bwd_row(const float*, const float*, const float*, const float*, float*, float*, int, int);
}
namespace simd_avx512 {
void gemm_acc(int, int, int, int, int, const float*, int, const float*, int, float*, int, float*);
void f32_to_bf16(const float*, unsigned short*, long long);
void bf16_to_f32(const unsigned short*, float*, long long);
bool f32_to_f16(const float*, unsigned short*, long long);
bool f16_to_f32(const unsigned short*, float*, long long);
void gelu_fwd(const float*, float*, long long);
void gelu_bwd(const float*, const float*, float*, long long);
void lstm_fwd_row(const float*, float*, const float*, float*, float*, int, int);
void lstm_bwd_row(const float*, const float*, const float*, const float*, float*, float*, int, int);
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

void lstm_fwd_row(const float* a, float* g, const float* c_prev, float* c, float* h, int n, int H) {
    if (g_level == 2) simd_avx512::lstm_fwd_row(a, g, c_prev, c, h, n, H);
    else if (g_level == 1) simd_avx2::lstm_fwd_row(a, g, c_prev, c, h, n, H);
    else simd_base::lstm_fwd_row(a, g, c_prev, c, h, n, H);
}

void lstm_bwd_row(const float* g, const float* c, const float* c_prev, const float* dh, float* da, float* dc,
                  int n, int H) {
    if (g_level == 2) simd_avx512::lstm_bwd_row(g, c, c_prev, dh, da, dc, n, H);
    else if (g_level == 1) simd_avx2::lstm_bwd_row(g, c, c_prev, dh, da, dc, n, H);
    else simd_base::lstm_bwd_row(g, c, c_prev, dh, da, dc, n, H);
}

// conversions: split over the threads, each chunk in the chosen build
#define SIMD_CALL(fn, ...) (g_level == 2 ? simd_avx512::fn(__VA_ARGS__) : g_level == 1 ? simd_avx2::fn(__VA_ARGS__) \
                                                                       : simd_base::fn(__VA_ARGS__))
void f32_to_bf16(const float* x, unsigned short* y, long long n) {
    split(n, [&](long long a, long long b) { SIMD_CALL(f32_to_bf16, x + a, y + a, b - a); });
}
void bf16_to_f32(const unsigned short* x, float* y, long long n) {
    split(n, [&](long long a, long long b) { SIMD_CALL(bf16_to_f32, x + a, y + a, b - a); });
}
bool f32_to_f16(const float* x, unsigned short* y, long long n) {
    if (g_level == 0) return false;
    split(n, [&](long long a, long long b) { SIMD_CALL(f32_to_f16, x + a, y + a, b - a); });
    return true;
}
bool f16_to_f32(const unsigned short* x, float* y, long long n) {
    if (g_level == 0) return false;
    split(n, [&](long long a, long long b) { SIMD_CALL(f16_to_f32, x + a, y + a, b - a); });
    return true;
}

// C += op(A) op(B) without BLAS: rows split over the threads (each its own
// packing buffers), or all on this thread inside a parallel region
void gemm_acc(bool ta, bool tb, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c,
              int ldc, bool threads) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    const long long work = (long long)m * n * k;
    auto rows = [&](int r0, int r1) {
        thread_local std::vector<float> buf;
        const size_t need = (size_t)72 * 256 + (size_t)256 * 1024;
        if (buf.size() < need) buf.resize(need);
        const float* ap = ta ? a + r0 : a + (long long)r0 * lda;
        SIMD_CALL(gemm_acc, ta ? 1 : 0, tb ? 1 : 0, r1 - r0, n, k, ap, lda, b, ldb, c + (long long)r0 * ldc, ldc,
                  buf.data());
    };
    bool par = threads && work >= (1LL << 21);       // measured (tests/bench_gemm.py): below, threads lose
#ifdef _OPENMP
    par = par && !omp_in_parallel();
#else
    par = false;
#endif
    if (!par) { rows(0, m); return; }
    #pragma omp parallel
    {
        long long t = 0, nt = 1;
#ifdef _OPENMP
        t = omp_get_thread_num();
        nt = omp_get_num_threads();
#endif
        const long long panels = (m + 5) / 6;
        if (panels >= nt) {
            // whole 6-row panels per thread
            const long long per = (panels + nt - 1) / nt;
            const int r0 = (int)std::min<long long>(m, t * per * 6), r1 = (int)std::min<long long>(m, (t + 1) * per * 6);
            if (r0 < r1) rows(r0, r1);
        } else {
            // too few rows to go round (32 x 1024 x 256: 6 panels for 16
            // threads ran at 29 GFLOP/s, MKL 231): 16-column blocks instead
            const long long blocks = (n + 15) / 16, per = (blocks + nt - 1) / nt;
            const int j0 = (int)std::min<long long>(n, t * per * 16), j1 = (int)std::min<long long>(n, (t + 1) * per * 16);
            if (j0 < j1) {
                thread_local std::vector<float> buf;
                const size_t need = (size_t)72 * 256 + (size_t)256 * 1024;
                if (buf.size() < need) buf.resize(need);
                const float* bp = tb ? b + (long long)j0 * ldb : b + j0;
                SIMD_CALL(gemm_acc, ta ? 1 : 0, tb ? 1 : 0, m, j1 - j0, k, a, lda, bp, ldb, c + j0, ldc, buf.data());
            }
        }
    }
}

} // namespace simd
