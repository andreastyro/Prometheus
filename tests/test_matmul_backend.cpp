// Verifies whichever matmul backend was compiled in against naive triple loops.
//
// matmul_forward and the two backward entry points are swapped at compile time
// between a tiled C++ kernel and cblas_sgemm. The BLAS calls pass explicit
// leading dimensions and transpose flags, and a mistake there produces correct
// results for square matrices while silently corrupting rectangular ones. So
// every shape below is deliberately non-square with odd, mutually prime sizes,
// and several are larger than the tile size so the tiled path takes its
// blocked branch rather than the remainder branch.
//
// This is the only test that exercises the BLAS code paths at all: builds with
// -DUSE_OPENBLAS=ON or -DUSE_MKL=ON route through cblas_sgemm, and without this
// nothing compared their output to a reference.

#include "ml/matmul_backend.hpp"
#include "ml/utils/testing.hpp"

#include <string>
#include <vector>

using namespace std;

// deterministic, mixed-sign, and not symmetric in any index
static vector<float> fill(int n, int salt) {
    vector<float> v(n);
    for (int i = 0; i < n; i++)
        v[i] = 0.137f * ((i * 7 + salt * 3) % 11 - 5) + 0.021f * ((i + salt) % 3);
    return v;
}

static void check_forward(int m, int n, int k) {
    vector<float> a = fill(m * k, 1), b = fill(k * n, 2);
    vector<float> c(m * n, 0.0f), ref(m * n, 0.0f);

    matmul_forward(m, n, k, a.data(), b.data(), c.data());

    // ref[i][j] = sum over kk of a[i][kk] * b[kk][j]
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int kk = 0; kk < k; kk++)
                acc += (double)a[i * k + kk] * (double)b[kk * n + j];
            ref[i * n + j] = (float)acc;
        }

    double worst = 0.0;
    for (int i = 0; i < m * n; i++) {
        double d = c[i] - ref[i];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
    }
    // tolerance scales with k: every output sums k products
    CHECK_NEAR(("forward " + to_string(m) + "x" + to_string(k) + " @ " +
                to_string(k) + "x" + to_string(n)).c_str(),
               worst, 0.0, 2e-4 * (double)k);
}

static void check_backward_a(int m, int inner, int b_cols) {
    vector<float> go = fill(m * b_cols, 3), bd = fill(inner * b_cols, 4);
    // the backward entries accumulate, so seed with something non-zero to
    // confirm they add rather than overwrite
    vector<float> ag = fill(m * inner, 5), ref = ag;

    matmul_backward_a(m, inner, b_cols, go.data(), bd.data(), ag.data());

    // dA += grad_out @ B^T
    for (int i = 0; i < m; i++)
        for (int j = 0; j < inner; j++) {
            double acc = 0.0;
            for (int kk = 0; kk < b_cols; kk++)
                acc += (double)go[i * b_cols + kk] * (double)bd[j * b_cols + kk];
            ref[i * inner + j] += (float)acc;
        }

    double worst = 0.0;
    for (int i = 0; i < m * inner; i++) {
        double d = ag[i] - ref[i];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
    }
    CHECK_NEAR(("backward_a " + to_string(m) + "x" + to_string(inner) +
                " (b_cols " + to_string(b_cols) + ")").c_str(),
               worst, 0.0, 2e-4 * (double)b_cols);
}

static void check_backward_b(int m, int inner, int b_cols) {
    vector<float> go = fill(m * b_cols, 6), ad = fill(m * inner, 7);
    vector<float> bg = fill(inner * b_cols, 8), ref = bg;

    matmul_backward_b(m, inner, b_cols, go.data(), ad.data(), bg.data());

    // dB += A^T @ grad_out
    for (int i = 0; i < inner; i++)
        for (int j = 0; j < b_cols; j++) {
            double acc = 0.0;
            for (int kk = 0; kk < m; kk++)
                acc += (double)ad[kk * inner + i] * (double)go[kk * b_cols + j];
            ref[i * b_cols + j] += (float)acc;
        }

    double worst = 0.0;
    for (int i = 0; i < inner * b_cols; i++) {
        double d = bg[i] - ref[i];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
    }
    CHECK_NEAR(("backward_b " + to_string(inner) + "x" + to_string(b_cols) +
                " (m " + to_string(m) + ")").c_str(),
               worst, 0.0, 2e-4 * (double)m);
}

int main() {
    printf("matmul backend: ");
#if defined(PROMETHEUS_USE_OPENBLAS)
    printf("OpenBLAS (cblas_sgemm)\n");
#elif defined(PROMETHEUS_USE_MKL)
    printf("Intel MKL (cblas_sgemm)\n");
#elif defined(PROMETHEUS_USE_OPENMP)
    printf("tiled + OpenMP\n");
#else
    printf("tiled, no external BLAS\n");
#endif

    // m, n, k chosen so no two are equal and none divide the tile size evenly
    const int shapes[][3] = {
        {1, 1, 1},      // degenerate
        {1, 5, 3},      // single row
        {5, 1, 3},      // single column
        {3, 4, 5},      // small rectangular
        {7, 3, 11},     // all distinct primes
        {13, 7, 5},
        {2, 9, 17},
        {33, 17, 9},    // past a 16-wide tile in one dim
        {65, 33, 17},   // past a 32/64 tile in all dims
        {64, 64, 64},   // exactly tile-aligned
        {100, 37, 63},  // large and awkward
    };

    printf("\n=== forward ===\n");
    for (auto& s : shapes) check_forward(s[0], s[1], s[2]);

    printf("\n=== backward wrt A ===\n");
    for (auto& s : shapes) check_backward_a(s[0], s[1], s[2]);

    printf("\n=== backward wrt B ===\n");
    for (auto& s : shapes) check_backward_b(s[0], s[1], s[2]);

    return test_report();
}
