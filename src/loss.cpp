#include "ml/loss.hpp"
#include "ml/ops.hpp"

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
    // log(pred)
    auto result = log_op(pred);

    result = multiply(target, result);
    result = sum(result, 1);
    result = mean(result, -1);
    result = multiply(-1.0f, result);

    return result;
}
