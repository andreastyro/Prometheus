#include "ml/loss.hpp"
#include "ml/ops.hpp"
#include "ml/autograd.hpp"
#include "ml/fastmath.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

TensorPtr mse_loss(TensorPtr pred, TensorPtr target){
    auto diff = subtract(pred, target);
    auto sq = pow_op(diff, 2.0f);
    return mean(sq, -1);
}

TensorPtr mae_loss(TensorPtr pred, TensorPtr target){
    auto diff = subtract(pred, target);
    auto ab = abs_op(diff);
    return mean(ab, -1);
}

TensorPtr bce_loss(TensorPtr pred, TensorPtr target){
    // A saturated sigmoid reaches exactly 0.0f or 1.0f in float, which makes
    // log(p) or log(1 - p) undefined (log_op throws). Clamp into (0, 1) first:
    // this is the standard numerically-safe BCE and costs nothing in accuracy.
    const float eps = 1e-7f;
    pred = clip(pred, eps, 1.0f - eps);

    // y * log(p)
    auto log_p = log_op(pred);
    auto term1 = multiply(target, log_p);

    // (1 - y) * log(1 - p)
    auto one_minus_pred = subtract(1.0f, pred);
    auto log_1_minus_p = log_op(one_minus_pred);
    auto one_minus_target = subtract(1.0f, target);
    auto term2 = multiply(one_minus_target, log_1_minus_p);

    // mean(-(term1 + term2))
    auto sum_terms = add(term1, term2);
    auto negated = multiply(-1.0f, sum_terms);
    return mean(negated, -1);
}

TensorPtr cross_entropy_loss(TensorPtr pred, TensorPtr target){
    auto result = log_op(pred);
    result = multiply(target, result);
    result = sum(result, 1);
    result = mean(result, -1);
    result = multiply(-1.0f, result);
    return result;
}

// Huber loss, mean over elements:
//   |r| <= delta :  0.5 * r^2          d/dpred = r / n
//   |r| >  delta :  delta*(|r| - d/2)  d/dpred = delta * sign(r) / n
// with r = pred - target. Previously this looped over raw floats and returned a
// detached scalar, so pred never received a gradient.
TensorPtr huber_loss(TensorPtr pred, TensorPtr target, float delta) {
    int n = pred->num_el();
    auto out = Tensor::zeros({1});
    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = std::abs(pred->data[i] - target->data[i]);
        if (diff <= delta)
            total += 0.5f * diff * diff;
        else
            total += delta * (diff - 0.5f * delta);
    }
    out->data[0] = total / n;

    if (pred->requires_grad) {
        auto node = make_node(out, {pred});
        node->backward_fn = [pred, target, out, delta, n]() {
            const float g = out->grad[0] / n;
            for (int i = 0; i < n; i++) {
                const float r = pred->data[i] - target->data[i];
                if (std::abs(r) <= delta)
                    pred->grad[i] += g * r;
                else
                    pred->grad[i] += g * delta * ((r > 0.0f) ? 1.0f : -1.0f);
            }
        };
    }

    return out;
}

TensorPtr kl_divergence(TensorPtr p, TensorPtr q) {
    // sum(p * log(p / q)), built from autograd ops so the gradient with
    // respect to p comes for free. The previous version looped over raw
    // floats and returned a detached tensor, so p never received a gradient.
    const float eps = 1e-8f;

    auto p_safe = clip(p, eps, 1.0f);
    auto q_safe = clip(q, eps, 1.0f);

    auto ratio = divide(p_safe, q_safe);
    auto terms = multiply(p_safe, log_op(ratio));

    // sum over every element, matching the previous reduction
    return multiply((float)terms->num_el(), mean(terms, -1));
}

TensorPtr reconstruction_loss(TensorPtr input, TensorPtr output) {
    // sum of squared differences — unnormalized MSE for autoencoders
    auto diff = subtract(input, output);
    return sum(multiply(diff, diff), -1);
}

// Triplet loss: max(0, d(a,p)^2 - d(a,n)^2 + margin).
// Inside the active region the gradients are
//   d/d anchor[i]   =  2*(dp - dn)   with dp = a-p, dn = a-n
//   d/d positive[i] = -2*dp
//   d/d negative[i] =  2*dn
// and all three are zero once the hinge is satisfied. Previously this returned
// a detached scalar, so none of the three inputs received a gradient.
TensorPtr contrastive_loss(TensorPtr anchor, TensorPtr positive, TensorPtr negative, float margin) {
    int n = anchor->num_el();
    float d_pos = 0.0f, d_neg = 0.0f;
    for (int i = 0; i < n; i++) {
        float dp = anchor->data[i] - positive->data[i];
        float dn = anchor->data[i] - negative->data[i];
        d_pos += dp * dp;
        d_neg += dn * dn;
    }
    auto out = Tensor::zeros({1});
    const float raw = d_pos - d_neg + margin;
    out->data[0] = std::max(0.0f, raw);

    if (anchor->requires_grad || positive->requires_grad || negative->requires_grad) {
        auto node = make_node(out, {anchor, positive, negative});
        node->backward_fn = [anchor, positive, negative, out, raw, n]() {
            if (raw <= 0.0f) return;            // hinge inactive, no gradient
            const float g = out->grad[0];
            for (int i = 0; i < n; i++) {
                const float dp = anchor->data[i] - positive->data[i];
                const float dn = anchor->data[i] - negative->data[i];
                if (anchor->requires_grad)   anchor->grad[i]   += g * 2.0f * (dp - dn);
                if (positive->requires_grad) positive->grad[i] -= g * 2.0f * dp;
                if (negative->requires_grad) negative->grad[i] += g * 2.0f * dn;
            }
        };
    }

    return out;
}

// L1 penalty: lambda * sum(|w|).  d/dw = lambda * sign(w)
// (sign(0) is taken as 0, the usual subgradient choice).
TensorPtr l1_regularization(std::vector<TensorPtr> params, float lambda_) {
    auto out = Tensor::zeros({1});
    float total = 0.0f;
    for (auto& p : params)
        for (float v : p->data)
            total += std::abs(v);
    out->data[0] = lambda_ * total;

    bool any_grad = false;
    for (auto& p : params) if (p->requires_grad) any_grad = true;

    if (any_grad) {
        auto node = make_node(out, params);
        node->backward_fn = [params, out, lambda_]() {
            const float g = out->grad[0];
            for (auto& p : params) {
                if (!p->requires_grad) continue;
                const int n_1 = p->num_el();
                for (int i = 0; i < n_1; i++) {
                    const float v = p->data[i];
                    const float sign = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
                    p->grad[i] += g * lambda_ * sign;
                }
            }
        };
    }

    return out;
}

// L2 penalty: lambda * sum(w^2).  d/dw = 2 * lambda * w
TensorPtr l2_regularization(std::vector<TensorPtr> params, float lambda_) {
    auto out = Tensor::zeros({1});
    float total = 0.0f;
    for (auto& p : params)
        for (float v : p->data)
            total += v * v;
    out->data[0] = lambda_ * total;

    bool any_grad = false;
    for (auto& p : params) if (p->requires_grad) any_grad = true;

    if (any_grad) {
        auto node = make_node(out, params);
        node->backward_fn = [params, out, lambda_]() {
            const float g = out->grad[0];
            for (auto& p : params) {
                if (!p->requires_grad) continue;
                const int n_2 = p->num_el();
                for (int i = 0; i < n_2; i++)
                    p->grad[i] += g * 2.0f * lambda_ * p->data[i];
            }
        };
    }

    return out;
}

// Sparse cross-entropy (single position).
// Numerically stable: loss = log(Σ exp(logits)) - logits[target]
//                          = log(Σ exp(logits - max)) + max - logits[target]
TensorPtr cross_entropy_sparse(TensorPtr logits, int target_idx) {
    int n = logits->num_el();
    float max_v = *std::max_element(logits->data.begin(), logits->data.end());

    float sum_exp = 0.0f;
    for (float v : logits->data)
        sum_exp += std::expf(v - max_v);

    float loss_val = std::logf(sum_exp) + max_v - logits->data[target_idx];

    auto result = make_shared<Tensor>(std::vector<int>{1});
    result->data[0] = loss_val;

    if (logits->requires_grad) {
        auto node = make_node(result, {logits});
        node->backward_fn = [logits, result, target_idx, n, max_v, sum_exp]() {
            float scale = result->grad[0];
            for (int j = 0; j < n; j++) {
                float p = std::expf(logits->data[j] - max_v) / sum_exp;
                logits->grad[j] += scale * (p - (j == target_idx ? 1.0f : 0.0f));
            }
        };
    }

    return result;
}

// Sparse cross-entropy over a full sequence — avoids building a loop in Python.
// logits: [seq_len, vocab_size],  targets: seq_len ints.
// Returns mean loss (scalar) with proper backward to all logit positions.
TensorPtr cross_entropy_sparse_seq(TensorPtr logits, const std::vector<int>& targets) {
    int seq_len  = logits->shape[0];
    int vocab    = logits->shape[1];

    // Forward: accumulate loss per position
    // 1024 positions x 16000 vocab is 16M exps per step, and this ran them
    // one at a time on one core. Rows are independent, so split them across
    // threads; within a row, exp into a scratch buffer first so that loop
    // vectorises, then sum it.
    std::vector<float> max_v(seq_len), sum_exp(seq_len, 0.0f);
    #pragma omp parallel if(seq_len >= 16)
    {
        std::vector<float> e(vocab);
        #pragma omp for schedule(static)
        for (int t = 0; t < seq_len; t++) {
            const float* row = logits->data.data() + (size_t)t * vocab;
            float mx = row[0];
            for (int v = 1; v < vocab; v++)
                mx = std::max(mx, row[v]);
            max_v[t] = mx;
            for (int v = 0; v < vocab; v++)
                e[v] = fastmath::exp(row[v] - mx);
            float s = 0.0f;
            for (int v = 0; v < vocab; v++)
                s += e[v];
            sum_exp[t] = s;
        }
    }

    float total = 0.0f;
    for (int t = 0; t < seq_len; t++)
        total += std::logf(sum_exp[t]) + max_v[t] - logits->data[t * vocab + targets[t]];

    auto result = make_shared<Tensor>(std::vector<int>{1});
    result->data[0] = total / (float)seq_len;

    if (logits->requires_grad) {
        auto node = make_node(result, {logits});
        node->backward_fn = [logits, result, targets, seq_len, vocab, max_v, sum_exp]() {
            const float scale = result->grad[0] / (float)seq_len;
            #pragma omp parallel for schedule(static) if(seq_len >= 16)
            for (int t = 0; t < seq_len; t++) {
                const float* row = logits->data.data() + (size_t)t * vocab;
                float* g = logits->grad.data() + (size_t)t * vocab;
                const float mx = max_v[t];
                const float k  = scale / sum_exp[t];
                // the target's -1 is applied once after the loop, so the loop
                // body has no branch and vectorises
                for (int v = 0; v < vocab; v++)
                    g[v] += k * fastmath::exp(row[v] - mx);
                g[targets[t]] -= scale;
            }
        };
    }

    return result;
}
