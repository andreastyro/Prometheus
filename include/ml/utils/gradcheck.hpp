#pragma once
/// Numeric gradient checking, in the spirit of torch.autograd.gradcheck.
///
/// Compares each analytical gradient produced by backward() against a central
/// finite difference of the forward pass:
///
///     d loss / d p  ~=  (loss(p + eps) - loss(p - eps)) / (2 * eps)
///
/// A severed computation graph shows up as an analytical gradient of exactly
/// zero where the numeric one is not -- the signature of a missing grad_fn.
///
/// Usage:
///     Linear fc(4, 3);
///     auto x = Tensor::randn({2, 4});
///     auto r = gradcheck([&]{ return fc.forward(x); }, fc.parameters());
///     if (!r.passed) printf("%s", r.report.c_str());
///
/// Note on eps and tolerances: the library computes in float32, and a central
/// difference of two nearly-equal O(1) values loses most of its significant
/// digits to cancellation. eps = 1e-2 with rtol = 2e-2 and atol = 1e-5 is the
/// usable band; much smaller eps is swamped by rounding, much larger picks up
/// the quadratic term of the Taylor expansion.

#include "ml/tensor.hpp"
#include <functional>
#include <string>
#include <vector>

struct GradCheckOptions {
    float eps        = 1e-2f;  ///< finite-difference step
    float atol       = 1e-5f;  ///< absolute tolerance (the float32 noise floor)
    float rtol       = 2e-2f;  ///< relative tolerance
    int   max_points = 4;      ///< elements probed per tensor (0 = all)
    bool  verbose    = false;  ///< print every comparison, not just failures
};

struct GradCheckResult {
    bool        passed      = true;
    int         checked     = 0;
    int         failed      = 0;
    double      max_abs_err = 0.0;
    double      max_rel_err = 0.0;
    std::string report;        ///< human-readable failure list (empty if passed)

    explicit operator bool() const { return passed; }
};

/// Check d(mean(forward()))/d(input) for every tensor in `inputs`.
/// `forward` must be re-runnable: it is called repeatedly with perturbed values.
GradCheckResult gradcheck(const std::function<TensorPtr()>& forward,
                          const std::vector<TensorPtr>& inputs,
                          const GradCheckOptions& opts = {});

/// Same, with a label used in the report. Convenient when checking many layers.
GradCheckResult gradcheck(const std::string& name,
                          const std::function<TensorPtr()>& forward,
                          const std::vector<TensorPtr>& inputs,
                          const GradCheckOptions& opts = {});
