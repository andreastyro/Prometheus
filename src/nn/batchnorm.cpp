#include "ml/nn/batchnorm.hpp"
#include "ml/tensor.hpp"
#include <cmath>
#include <stdexcept>
#include <string>   // to_string; MSVC does not get it transitively
#include <vector>

using namespace std;

BatchNorm::BatchNorm(int num_features, float eps, bool training, float momentum)
    : eps(eps), training(training), momentum(momentum),
      running_mean(num_features, 0.0f), running_var(num_features, 1.0f) {
    gamma = Tensor::ones({num_features});
    beta  = Tensor::zeros({num_features});
    gamma->requires_grad = true;
    beta->requires_grad  = true;
}

// Statistics are per feature/channel, taken over the batch and, for conv input,
// over the spatial positions too. Viewing the tensor as [N, C, S]:
//
//   2D [batch, features]          -> S = 1,    the BatchNorm1d case
//   4D [batch, channels, H, W]    -> S = H*W,  the BatchNorm2d case
//
// A 4D input used to be read as if it were 2D: shape[1] was taken as the
// feature count and elements indexed b * features + f, which normalised the
// wrong groups and returned silently wrong numbers rather than erroring. Any
// CNN using BatchNorm was affected.
//
// Backward is the usual normalisation gradient, over N*S elements per channel:
//
//   dL/dx[i] = (1/std) * ( g[i] - mean(g) - x_hat[i] * mean(g * x_hat) )
//
// with g[i] = gamma[c] * dL/dy[i]. In eval mode the statistics come from the
// running estimate and are constants, so the last two terms vanish.
TensorPtr BatchNorm::forward(TensorPtr input) {
    const int ndim = (int)input->shape.size();
    if (ndim < 2)
        throw runtime_error("BatchNorm: expected at least 2 dimensions "
                            "[batch, features, ...]");

    const int N = input->shape[0];
    const int C = input->shape[1];
    if (C != (int)running_mean.size())
        throw runtime_error("BatchNorm: input has " + to_string(C) +
                            " channels but the layer was built for " +
                            to_string(running_mean.size()));

    int S = 1;
    for (int d = 2; d < ndim; d++) S *= input->shape[d];

    auto output = make_shared<Tensor>(input->shape);

    // saved for the backward pass
    vector<float> x_hat(input->num_el());
    vector<float> inv_std_all(C);

    const int count = N * S;

    for (int c = 0; c < C; c++) {
        float mean, var;

        if (training) {
            mean = 0.0f;
            for (int n = 0; n < N; n++)
                for (int s = 0; s < S; s++)
                    mean += input->data[(n * C + c) * S + s];
            mean /= count;

            var = 0.0f;
            for (int n = 0; n < N; n++)
                for (int s = 0; s < S; s++) {
                    float diff = input->data[(n * C + c) * S + s] - mean;
                    var += diff * diff;
                }
            var /= count;

            // exponential moving average, as PyTorch does
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var;
        } else {
            mean = running_mean[c];
            var  = running_var[c];
        }

        const float inv_std = 1.0f / std::sqrt(var + eps);
        inv_std_all[c] = inv_std;

        for (int n = 0; n < N; n++)
            for (int s = 0; s < S; s++) {
                const int idx = (n * C + c) * S + s;
                const float norm = (input->data[idx] - mean) * inv_std;
                x_hat[idx] = norm;
                output->data[idx] = gamma->data[c] * norm + beta->data[c];
            }
    }

    if (input->requires_grad || gamma->requires_grad || beta->requires_grad) {
        auto node = make_node(output, {input, gamma, beta});

        auto gamma_ = gamma;
        auto beta_  = beta;
        const bool was_training = training;

        node->backward_fn = [input, output, gamma_, beta_, x_hat, inv_std_all,
                             N, C, S, count, was_training]() {
            for (int c = 0; c < C; c++) {
                const float inv_std = inv_std_all[c];

                float sum_g = 0.0f, sum_g_xhat = 0.0f;
                if (was_training) {
                    for (int n = 0; n < N; n++)
                        for (int s = 0; s < S; s++) {
                            const int idx = (n * C + c) * S + s;
                            const float gi = gamma_->data[c] * output->grad[idx];
                            sum_g      += gi;
                            sum_g_xhat += gi * x_hat[idx];
                        }
                }

                const float mean_g      = was_training ? sum_g / count : 0.0f;
                const float mean_g_xhat = was_training ? sum_g_xhat / count : 0.0f;

                for (int n = 0; n < N; n++)
                    for (int s = 0; s < S; s++) {
                        const int idx = (n * C + c) * S + s;
                        const float gi = gamma_->data[c] * output->grad[idx];

                        if (input->requires_grad) {
                            if (was_training)
                                input->grad[idx] +=
                                    inv_std * (gi - mean_g - x_hat[idx] * mean_g_xhat);
                            else
                                input->grad[idx] += inv_std * gi;
                        }

                        if (gamma_->requires_grad)
                            gamma_->grad[c] += output->grad[idx] * x_hat[idx];
                        if (beta_->requires_grad)
                            beta_->grad[c] += output->grad[idx];
                    }
            }
        };
    }

    return output;
}

vector<TensorPtr> BatchNorm::parameters() {
    return {gamma, beta};
}
