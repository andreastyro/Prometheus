#pragma once
// Minimal assertion helpers so test binaries FAIL (non-zero exit) instead of
// printing a number next to the word "expected" and exiting 0 regardless.
//
// Usage:
//     #include "ml/utils/testing.hpp"
//     int main() {
//         CHECK_NEAR("pos 1 dim 0", pe->data[4], 0.8415f, 1e-4f);
//         CHECK_TRUE("output is finite", all_finite(out->data));
//         return test_report();
//     }

#include <cstdio>
#include <cmath>

inline int  g_test_checks   = 0;
inline int  g_test_failures = 0;

inline void test_check_near(const char* what, double actual, double expected, double tol) {
    g_test_checks++;
    const bool ok = std::fabs(actual - expected) <= tol;
    if (!ok) g_test_failures++;
    std::printf("  %-52s %14.6f  expected %14.6f   %s\n",
                what, actual, expected, ok ? "ok" : "** FAIL **");
}

inline void test_check_true(const char* what, bool cond) {
    g_test_checks++;
    if (!cond) g_test_failures++;
    std::printf("  %-52s %14s                           %s\n",
                what, cond ? "true" : "false", cond ? "ok" : "** FAIL **");
}

// Combined absolute + relative check: pass if
//     |actual - expected| <= abs_tol + rel_tol * |expected|
// This is the criterion numpy.allclose and torch.autograd.gradcheck use. The
// absolute floor matters for finite differences: computing
// (f(x+eps) - f(x-eps)) / 2*eps in float32 loses most of its significant
// digits to cancellation, leaving a noise floor near 1e-5 for O(1) losses.
// Without it, two gradients that are both ~1e-7 differ by 100% relatively and
// report a failure that carries no information.
inline void test_check_rel(const char* what, double actual, double expected,
                           double rel_tol, double abs_tol = 1e-5) {
    g_test_checks++;
    const bool ok = std::fabs(actual - expected) <= abs_tol + rel_tol * std::fabs(expected);
    if (!ok) g_test_failures++;
    std::printf("  %-52s %14.6f  expected %14.6f   %s\n",
                what, actual, expected, ok ? "ok" : "** FAIL **");
}

inline int test_report() {
    std::printf("\n%d checks, %d failed\n", g_test_checks, g_test_failures);
    if (g_test_failures == 0) std::printf("PASS\n");
    else                      std::printf("FAIL\n");
    return g_test_failures == 0 ? 0 : 1;
}

#define CHECK_NEAR(what, actual, expected, tol) test_check_near((what), (actual), (expected), (tol))
#define CHECK_REL(what, actual, expected, tol)  test_check_rel((what), (actual), (expected), (tol))
#define CHECK_TRUE(what, cond)                  test_check_true((what), (cond))
