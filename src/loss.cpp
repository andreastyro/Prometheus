#include "ml/loss.hpp"
#include "ml/ops.hpp"
#include <cmath>

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
