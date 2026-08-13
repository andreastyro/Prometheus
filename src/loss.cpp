#include "ml/loss.hpp"
#include "ml/ops.hpp"
#include "ml/autograd.hpp"
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
    return out;
}

TensorPtr kl_divergence(TensorPtr p, TensorPtr q) {
    // sum(p * log(p / (q + eps))) — eps avoids log(0)
    int n = p->num_el();
    auto out = Tensor::zeros({1});
    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        float pi = p->data[i];
        float qi = q->data[i] + 1e-8f;
        if (pi > 0.0f)
            total += pi * std::log(pi / qi);
    }
    out->data[0] = total;
    return out;
}

TensorPtr reconstruction_loss(TensorPtr input, TensorPtr output) {
    // sum of squared differences — unnormalized MSE for autoencoders
    auto diff = subtract(input, output);
    return sum(multiply(diff, diff), -1);
}

TensorPtr contrastive_loss(TensorPtr anchor, TensorPtr positive, TensorPtr negative, float margin) {
    // triplet loss: max(0, d(a,p)^2 - d(a,n)^2 + margin)
    int n = anchor->num_el();
    float d_pos = 0.0f, d_neg = 0.0f;
    for (int i = 0; i < n; i++) {
        float dp = anchor->data[i] - positive->data[i];
        float dn = anchor->data[i] - negative->data[i];
        d_pos += dp * dp;
        d_neg += dn * dn;
    }
    auto out = Tensor::zeros({1});
    out->data[0] = std::max(0.0f, d_pos - d_neg + margin);
    return out;
}

TensorPtr l1_regularization(std::vector<TensorPtr> params, float lambda_) {
    auto out = Tensor::zeros({1});
    float total = 0.0f;
    for (auto& p : params)
        for (float v : p->data)
            total += std::abs(v);
    out->data[0] = lambda_ * total;
    return out;
}

TensorPtr l2_regularization(std::vector<TensorPtr> params, float lambda_) {
    auto out = Tensor::zeros({1});
    float total = 0.0f;
    for (auto& p : params)
        for (float v : p->data)
            total += v * v;
    out->data[0] = lambda_ * total;
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
    std::vector<float> max_v(seq_len), sum_exp(seq_len, 0.0f);
    for (int t = 0; t < seq_len; t++) {
        float mx = logits->data[t * vocab];
        for (int v = 1; v < vocab; v++)
            mx = std::max(mx, logits->data[t * vocab + v]);
        max_v[t] = mx;
        for (int v = 0; v < vocab; v++)
            sum_exp[t] += std::expf(logits->data[t * vocab + v] - mx);
    }

    float total = 0.0f;
    for (int t = 0; t < seq_len; t++)
        total += std::logf(sum_exp[t]) + max_v[t] - logits->data[t * vocab + targets[t]];

    auto result = make_shared<Tensor>(std::vector<int>{1});
    result->data[0] = total / (float)seq_len;

    if (logits->requires_grad) {
        auto node = make_node(result, {logits});
        node->backward_fn = [logits, result, targets, seq_len, vocab, max_v, sum_exp]() {
            float scale = result->grad[0] / (float)seq_len;
            for (int t = 0; t < seq_len; t++) {
                for (int v = 0; v < vocab; v++) {
                    float p = std::expf(logits->data[t * vocab + v] - max_v[t]) / sum_exp[t];
                    logits->grad[t * vocab + v] += scale * (p - (v == targets[t] ? 1.0f : 0.0f));
                }
            }
        };
    }

    return result;
}
