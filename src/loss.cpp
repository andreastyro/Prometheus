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
        float diff = std::abs(pred->data()[i] - target->data()[i]);
        if (diff <= delta)
            total += 0.5f * diff * diff;
        else
            total += delta * (diff - 0.5f * delta);
    }
    out->data()[0] = total / n;

    if (pred->requires_grad()) {
        auto node = make_node(out, {pred});
        node->name = "huber_loss";
        autograd::attach_reference(*node, [tgt = target, delta](const std::vector<TensorPtr>& in) -> TensorPtr {
            // quadratic inside |d| <= delta, linear outside; which side each element
            // is on is fixed at this point
            auto d = subtract(in[0], tgt);
            auto inside = make_shared<Tensor>(d->shape());
            {
                const float* D = d->data();
                float* M = inside->data();
                for (int64_t i = 0; i < d->numel(); i++) M[i] = std::abs(D[i]) <= delta ? 1.0f : 0.0f;
            }
            auto quad = multiply(0.5f, multiply(d, d));
            auto lin  = multiply(delta, subtract(abs_op(d), 0.5f * delta));
            return mean(add(multiply(quad, inside), multiply(lin, subtract(1.0f, inside))), -1);
        });
        node->backward_fn = [pred, target, delta, n](const TensorPtr& out) {
            const float g = out->grad()[0] / n;
            for (int i = 0; i < n; i++) {
                const float r = pred->data()[i] - target->data()[i];
                if (std::abs(r) <= delta)
                    pred->grad()[i] += g * r;
                else
                    pred->grad()[i] += g * delta * ((r > 0.0f) ? 1.0f : -1.0f);
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
        float dp = anchor->data()[i] - positive->data()[i];
        float dn = anchor->data()[i] - negative->data()[i];
        d_pos += dp * dp;
        d_neg += dn * dn;
    }
    auto out = Tensor::zeros({1});
    const float raw = d_pos - d_neg + margin;
    out->data()[0] = std::max(0.0f, raw);

    if (anchor->requires_grad() || positive->requires_grad() || negative->requires_grad()) {
        auto node = make_node(out, {anchor, positive, negative});
        node->name = "contrastive_loss";
        autograd::attach_reference(*node, [margin, active = raw > 0.0f](const std::vector<TensorPtr>& in) -> TensorPtr {
            // max(0, |a - p|^2 - |a - n|^2 + margin), the hinge's side fixed
            auto dp = subtract(in[0], in[1]);
            auto dn = subtract(in[0], in[2]);
            auto raw_t = add(margin, subtract(sum(multiply(dp, dp), -1), sum(multiply(dn, dn), -1)));
            return multiply(active ? 1.0f : 0.0f, raw_t);
        });
        node->backward_fn = [anchor, positive, negative, raw, n](const TensorPtr& out) {
            if (raw <= 0.0f) return;            // hinge inactive, no gradient
            const float g = out->grad()[0];
            for (int i = 0; i < n; i++) {
                const float dp = anchor->data()[i] - positive->data()[i];
                const float dn = anchor->data()[i] - negative->data()[i];
                if (anchor->requires_grad())   anchor->grad()[i]   += g * 2.0f * (dp - dn);
                if (positive->requires_grad()) positive->grad()[i] -= g * 2.0f * dp;
                if (negative->requires_grad()) negative->grad()[i] += g * 2.0f * dn;
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
        for (float v : p->values())
            total += std::abs(v);
    out->data()[0] = lambda_ * total;

    bool any_grad = false;
    for (auto& p : params) if (p->requires_grad()) any_grad = true;

    if (any_grad) {
        auto node = make_node(out, params);
        node->name = "l1_regularization";
        autograd::attach_reference(*node, [lambda_](const std::vector<TensorPtr>& in) -> TensorPtr {
            TensorPtr total;
            for (auto& t : in) { auto s = sum(abs_op(t), -1); total = total ? add(total, s) : s; }
            return multiply(lambda_, total);
        });
        node->backward_fn = [params, lambda_](const TensorPtr& out) {
            const float g = out->grad()[0];
            for (auto& p : params) {
                if (!p->requires_grad()) continue;
                const int n_1 = p->num_el();
                for (int i = 0; i < n_1; i++) {
                    const float v = p->data()[i];
                    const float sign = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
                    p->grad()[i] += g * lambda_ * sign;
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
        for (float v : p->values())
            total += v * v;
    out->data()[0] = lambda_ * total;

    bool any_grad = false;
    for (auto& p : params) if (p->requires_grad()) any_grad = true;

    if (any_grad) {
        auto node = make_node(out, params);
        node->name = "l2_regularization";
        autograd::attach_reference(*node, [lambda_](const std::vector<TensorPtr>& in) -> TensorPtr {
            TensorPtr total;
            for (auto& t : in) { auto s = sum(multiply(t, t), -1); total = total ? add(total, s) : s; }
            return multiply(lambda_, total);
        });
        node->backward_fn = [params, lambda_](const TensorPtr& out) {
            const float g = out->grad()[0];
            for (auto& p : params) {
                if (!p->requires_grad()) continue;
                const int n_2 = p->num_el();
                for (int i = 0; i < n_2; i++)
                    p->grad()[i] += g * 2.0f * lambda_ * p->data()[i];
            }
        };
    }

    return out;
}

// Sparse cross-entropy (single position).
// Numerically stable: loss = log(Σ exp(logits)) - logits[target]
//                          = log(Σ exp(logits - max)) + max - logits[target]
TensorPtr cross_entropy_sparse(TensorPtr logits, int target_idx) {
    // max_element over ONE buffer: this read begin() and end() from two
    // separate temporary copies, which is undefined behaviour
    logits = checked_input("cross_entropy_sparse", logits);
    int n = logits->num_el();
    auto vals = logits->values();
    float max_v = *std::max_element(vals.begin(), vals.end());

    float sum_exp = 0.0f;
    for (float v : logits->values())
        sum_exp += std::exp(v - max_v);

    float loss_val = std::log(sum_exp) + max_v - logits->data()[target_idx];

    auto result = make_shared<Tensor>(Shape{1});
    result->data()[0] = loss_val;

    if (logits->requires_grad()) {
        auto node = make_node(result, {logits});
        node->name = "cross_entropy_sparse";
        autograd::attach_reference(*node, [target_idx](const std::vector<TensorPtr>& in) -> TensorPtr {
            // -log softmax(logits)[target]
            auto flat = reshape(in[0], {in[0]->numel()});
            auto pick = make_shared<vector<int64_t>>(1, (int64_t)target_idx);
            return multiply(-1.0f, log_op(gather_flat(softmax(flat), pick, {1})));
        });
        node->backward_fn = [logits, target_idx, n, max_v, sum_exp](const TensorPtr& result) {
            float scale = result->grad()[0];
            for (int j = 0; j < n; j++) {
                float p = std::exp(logits->data()[j] - max_v) / sum_exp;
                logits->grad()[j] += scale * (p - (j == target_idx ? 1.0f : 0.0f));
            }
        };
    }

    return result;
}

// Sparse cross-entropy over a full sequence — avoids building a loop in Python.
// logits: [seq_len, vocab_size],  targets: seq_len ints.
// Returns mean loss (scalar) with proper backward to all logit positions.
TensorPtr cross_entropy_sparse_seq(TensorPtr logits, const std::vector<int>& targets) {
    // logits are [..., vocab]: every leading dimension is a position
    logits = checked_input("cross_entropy_sparse_seq", logits);
    int vocab    = (int)logits->shape().back();
    int seq_len  = (int)(logits->numel() / vocab);
    if ((int)targets.size() != seq_len)
        throw runtime_error("cross_entropy_sparse_seq: " + to_string(targets.size()) +
                            " targets for " + to_string(seq_len) + " positions");

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
            const float* row = logits->data() + (size_t)t * vocab;
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
        total += std::log(sum_exp[t]) + max_v[t] - logits->data()[t * vocab + targets[t]];

    auto result = make_shared<Tensor>(Shape{1});
    result->data()[0] = total / (float)seq_len;

    if (logits->requires_grad()) {
        auto node = make_node(result, {logits});
        node->backward_graph = [logits, targets, seq_len, vocab](const TensorPtr&, const TensorPtr& G) -> std::vector<TensorPtr> {
            // (softmax(logits) - onehot(targets)) / positions, scaled by the upstream gradient
            auto onehot = make_shared<Tensor>(logits->shape());
            for (int t = 0; t < seq_len; t++) onehot->data()[(size_t)t * vocab + targets[t]] = 1.0f;
            auto d = multiply(1.0f / (float)seq_len, subtract(softmax(logits), onehot));
            return {multiply(reshape(G, {1}), d)};
        };
        node->name = "cross_entropy_sparse_seq";
        node->backward_fn = [logits, targets, seq_len, vocab, max_v, sum_exp](const TensorPtr& result) {
            const float scale = result->grad()[0] / (float)seq_len;
            #pragma omp parallel for schedule(static) if(seq_len >= 16)
            for (int t = 0; t < seq_len; t++) {
                const float* row = logits->data() + (size_t)t * vocab;
                float* g = logits->grad() + (size_t)t * vocab;
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
