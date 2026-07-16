#include "ml/tensor.hpp"
#include "ml/matmul_backend.hpp"
#include <chrono>
#include <cstdio>
#include <vector>
#include <cmath>

using namespace std;
using clock_t2 = chrono::high_resolution_clock;

// ── Original naive matmul (ijk order) ────────────────────────────────────────
static void matmul_naive(int m, int n, int k,
                         const float* a, const float* b, float* c) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int kk = 0; kk < k; kk++)
                c[i * n + j] += a[i * k + kk] * b[kk * n + j];
}

// ── Tiled cache-friendly matmul (always compiled in for comparison) ───────────
static constexpr int BENCH_TILE = 64;
static void matmul_tiled(int m, int n, int k,
                         const float* a, const float* b, float* c) {
    for (int i0 = 0; i0 < m; i0 += BENCH_TILE)
        for (int k0 = 0; k0 < k; k0 += BENCH_TILE)
            for (int j0 = 0; j0 < n; j0 += BENCH_TILE) {
                int i_end = std::min(i0 + BENCH_TILE, m);
                int k_end = std::min(k0 + BENCH_TILE, k);
                int j_end = std::min(j0 + BENCH_TILE, n);
                for (int i = i0; i < i_end; i++)
                    for (int kk = k0; kk < k_end; kk++) {
                        float a_ik = a[i * k + kk];
                        for (int j = j0; j < j_end; j++)
                            c[i * n + j] += a_ik * b[kk * n + j];
                    }
            }
}

// ── Timing helper — returns milliseconds ─────────────────────────────────────
static double time_ms(int m, int n, int k,
                      void (*fn)(int, int, int, const float*, const float*, float*),
                      const vector<float>& a, const vector<float>& b,
                      int reps = 5) {
    vector<float> c(m * n, 0.0f);
    // warmup
    fn(m, n, k, a.data(), b.data(), c.data());

    auto t0 = clock_t2::now();
    for (int r = 0; r < reps; r++) {
        fill(c.begin(), c.end(), 0.0f);
        fn(m, n, k, a.data(), b.data(), c.data());
    }
    auto t1 = clock_t2::now();
    return chrono::duration<double, milli>(t1 - t0).count() / reps;
}

// ── Correctness check ─────────────────────────────────────────────────────────
static bool results_match(int m, int n, int k,
                          const vector<float>& a, const vector<float>& b) {
    vector<float> c1(m * n, 0.0f), c2(m * n, 0.0f);
    matmul_naive(m, n, k, a.data(), b.data(), c1.data());
    matmul_forward(m, n, k, a.data(), b.data(), c2.data());
    for (int i = 0; i < m * n; i++)
        if (fabs(c1[i] - c2[i]) > 1e-3f) return false;
    return true;
}

// ── Fill with deterministic values ───────────────────────────────────────────
static vector<float> make_matrix(int rows, int cols, float seed = 1.0f) {
    vector<float> m(rows * cols);
    for (int i = 0; i < rows * cols; i++)
        m[i] = seed * (float)(i % 17 - 8) * 0.1f;
    return m;
}

int main() {
#if defined(PROMETHEUS_USE_OPENBLAS)
    const char* backend = "OpenBLAS";
#elif defined(PROMETHEUS_USE_MKL)
    const char* backend = "Intel MKL";
#elif defined(PROMETHEUS_USE_OPENMP)
    const char* backend = "tiled+OpenMP";
#else
    const char* backend = "(tiled only)";
#endif

    printf("Matmul benchmark — three-way comparison\n");
    printf("Backend: %s\n", backend);
    printf("========================================================\n\n");

    struct Case { int m, n, k; const char* label; };
    Case cases[] = {
        {  32,  32,  32, " 32x32 "},
        { 128, 128, 128, "128x128"},
        { 256, 256, 256, "256x256"},
        { 512, 512, 512, "512x512"},
        {1024,1024,1024, "1024x1024"},
    };

    printf("%-12s  %10s  %10s  %12s  %8s  %8s\n",
           "size", "naive(ms)", "tiled(ms)", backend, "vs naive", "vs tiled");
    printf("%-12s  %10s  %10s  %12s  %8s  %8s\n",
           "----", "---------", "---------", "------------", "--------", "--------");

    for (auto& c : cases) {
        auto a = make_matrix(c.m, c.k, 1.0f);
        auto b = make_matrix(c.k, c.n, 0.7f);

        double t_naive = time_ms(c.m, c.n, c.k, matmul_naive,   a, b);
        double t_tiled = time_ms(c.m, c.n, c.k, matmul_tiled,   a, b);
        double t_fast  = time_ms(c.m, c.n, c.k, matmul_forward, a, b);

        printf("%-12s  %10.3f  %10.3f  %12.3f  %7.1fx  %7.1fx\n",
               c.label,
               t_naive, t_tiled, t_fast,
               t_naive / t_fast,
               t_tiled / t_fast);
    }

    printf("\n");
    return 0;
}
