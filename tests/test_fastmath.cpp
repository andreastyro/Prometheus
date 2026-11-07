// fastmath::exp and fastmath::tanh replace libm in softmax, GELU and the
// cross-entropy loss. Those paths are checked end to end against numpy, but
// only at the inputs those tests happen to use. This sweeps the whole range.
#include "ml/fastmath.hpp"
#include <cmath>
#include <cstdio>

static int failures = 0;

static void check(const char* name, double worst, double limit) {
    const bool ok = worst <= limit;
    std::printf("  %-44s worst %.3e  (limit %.0e)  %s\n",
                name, worst, limit, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

int main() {
    std::printf("\n=== fastmath ===\n\n");

    // exp: relative error across the range a softmax actually sees
    double worst = 0.0;
    for (int i = 0; i <= 2000000; i++) {
        const float x = -80.0f + 160.0f * (float)i / 2000000.0f;
        const double ref = std::exp((double)x);
        const double got = fastmath::exp(x);
        worst = std::fmax(worst, std::fabs(got - ref) / ref);
    }
    check("exp relative error on [-80, 80]", worst, 5e-7);

    // exp of exactly 0 is 1: every softmax row has one
    check("exp(0) == 1", std::fabs(fastmath::exp(0.0f) - 1.0), 0.0);

    // a masked score is -1e9 minus the row max; it must come out as nothing
    check("exp(-1e9) is negligible", fastmath::exp(-1e9f), 1e-37);

    // and the top end saturates instead of overflowing to inf
    check("exp(1e9) is finite", std::isfinite(fastmath::exp(1e9f)) ? 0.0 : 1.0, 0.0);

    // tanh: absolute error, which is what GELU's 0.5x(1 + tanh) sees
    worst = 0.0;
    for (int i = 0; i <= 2000000; i++) {
        const float x = -20.0f + 40.0f * (float)i / 2000000.0f;
        worst = std::fmax(worst, std::fabs(fastmath::tanh(x) - std::tanh((double)x)));
    }
    check("tanh absolute error on [-20, 20]", worst, 1e-6);

    check("tanh(+50) == 1",  std::fabs(fastmath::tanh( 50.0f) - 1.0), 0.0);
    check("tanh(-50) == -1", std::fabs(fastmath::tanh(-50.0f) + 1.0), 0.0);

    std::printf("\n%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
