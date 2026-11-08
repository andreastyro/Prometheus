#include "ml/utils/gradcheck.hpp"
#include "ml/ops.hpp"

#include <cmath>
#include <cstdio>

using namespace std;

// Reduce any output to a single scalar so there is one loss to differentiate.
static TensorPtr scalarise(TensorPtr out) { return mean(out, -1); }

GradCheckResult gradcheck(const function<TensorPtr()>& forward,
                          const vector<TensorPtr>& inputs,
                          const GradCheckOptions& opts) {
    GradCheckResult r;
    char line[320];

    for (size_t t = 0; t < inputs.size(); t++) {
        TensorPtr p = inputs[t];
        if (!p || p->num_el() == 0) continue;

        // A tensor that does not require grad has no analytical gradient to
        // compare against; skipping it silently would hide a missing flag, so
        // report it instead.
        if (!p->requires_grad()) {
            snprintf(line, sizeof(line),
                     "  tensor %zu: requires_grad is false, nothing to check\n", t);
            r.report += line;
            continue;
        }

        // analytical gradients
        for (const TensorPtr& q : inputs) if (q) q->reset_grad();
        auto loss = scalarise(forward());

        // A forward that produced no graph cannot be differentiated. Report it
        // as the detachment it is, rather than letting backward() throw out of
        // a diagnostic routine.
        if (!loss->grad_fn()) {
            snprintf(line, sizeof(line),
                     "  tensor %zu: the forward pass produced no computation "
                     "graph, so every analytical gradient is zero\n", t);
            r.report += line;
            r.passed = false;
            r.checked++;
            r.failed++;
            continue;
        }

        loss->backward();
        vector<float> analytic = p->grad_vector();

        const int n    = p->num_el();
        int step = 1;
        if (opts.max_points > 0 && n > opts.max_points)
            step = (n + opts.max_points - 1) / opts.max_points;

        for (int i = 0; i < n; i += step) {
            const float orig = p->data()[i];

            // A central difference over the step actually taken: orig + h is
            // rounded to float32, so dividing by 2h was itself an error of up
            // to an ulp of orig over h.
            auto central = [&](float h) -> double {
                const float up = orig + h, dn = orig - h;
                p->data()[i] = up;
                const double lp = scalarise(forward())->data()[0];
                p->data()[i] = dn;
                const double lm = scalarise(forward())->data()[0];
                p->data()[i] = orig;
                return (lp - lm) / ((double)up - (double)dn);
            };

            // Richardson extrapolation. A central difference is off by
            // c*h^2 + O(h^4); combining steps h and h/2 cancels the h^2 term.
            // Without it a sharp function -- attention's softmax, ten steps of
            // a recurrence -- left truncation error of several percent at the
            // default step, failing correct gradients depending on which random
            // weights a test happened to draw. A smaller step would trade that
            // for float32 rounding; this removes it instead.
            const double numeric = (4.0 * central(opts.eps * 0.5f) - central(opts.eps)) / 3.0;
            const double got     = (i < (int)analytic.size()) ? analytic[i] : 0.0;

            const double abs_err = fabs(got - numeric);
            const double rel_err = abs_err / fmax(1e-12, fabs(numeric));
            if (abs_err > r.max_abs_err) r.max_abs_err = abs_err;
            if (abs_err > opts.atol && rel_err > r.max_rel_err) r.max_rel_err = rel_err;

            const bool ok = abs_err <= opts.atol + opts.rtol * fabs(numeric);
            r.checked++;
            if (!ok) {
                r.failed++;
                r.passed = false;
                snprintf(line, sizeof(line),
                         "  tensor %zu [%d]: analytic %.8f  numeric %.8f  "
                         "abs %.2e  rel %.2e\n",
                         t, i, got, numeric, abs_err, rel_err);
                r.report += line;
            } else if (opts.verbose) {
                printf("  tensor %zu [%d]: analytic %.8f  numeric %.8f  ok\n",
                       t, i, got, numeric);
            }
        }
    }
    return r;
}

GradCheckResult gradcheck(const string& name,
                          const function<TensorPtr()>& forward,
                          const vector<TensorPtr>& inputs,
                          const GradCheckOptions& opts) {
    GradCheckResult r = gradcheck(forward, inputs, opts);
    if (!r.report.empty()) r.report = name + ":\n" + r.report;
    return r;
}
