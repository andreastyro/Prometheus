#include "ml/nn/batchnorm.hpp"
#include "ml/tensor.hpp"
#include <cmath>
#include <vector>

using namespace std;

BatchNorm::BatchNorm(int num_features, float eps, bool training)
    : eps(eps), training(training) {
    gamma = Tensor::ones({num_features});
    beta  = Tensor::zeros({num_features});
    gamma->requires_grad = true;
    beta->requires_grad  = true;
}

// Same normalisation backward as LayerNorm and GroupNorm, except the statistics
// are taken down the batch axis for each feature independently. With N = batch
// and g[b] = gamma[f] * dL/dy[b, f]:
//
//     dL/dx[b, f] = (1/std) * ( g[b] - mean(g) - x_hat[b, f] * mean(g * x_hat) )
TensorPtr BatchNorm::forward(TensorPtr input) {
    // input: [batch, num_features]
    int batch        = input->shape[0];
    int num_features = input->shape[1];
    auto output      = make_shared<Tensor>(input->shape);

    // saved for the backward pass
    vector<float> x_hat(input->num_el());
    vector<float> inv_std_all(num_features);

    for (int f = 0; f < num_features; f++) {
        // mean over batch for this feature
        float mean = 0.0f;
        for (int b = 0; b < batch; b++)
            mean += input->data[b * num_features + f];
        mean /= batch;

        // variance over batch
        float var = 0.0f;
        for (int b = 0; b < batch; b++) {
            float diff = input->data[b * num_features + f] - mean;
            var += diff * diff;
        }
        var /= batch;

        float inv_std = 1.0f / std::sqrt(var + eps);
        inv_std_all[f] = inv_std;

        for (int b = 0; b < batch; b++) {
            int idx = b * num_features + f;
            float norm = (input->data[idx] - mean) * inv_std;
            x_hat[idx] = norm;
            output->data[idx] = gamma->data[f] * norm + beta->data[f];
        }
    }

    if (input->requires_grad || gamma->requires_grad || beta->requires_grad) {
        auto node = make_node(output, {input, gamma, beta});

        auto gamma_ = gamma;
        auto beta_  = beta;

        node->backward_fn = [input, output, gamma_, beta_, x_hat, inv_std_all,
                             batch, num_features]() {
            const int N = batch;

            for (int f = 0; f < num_features; f++) {
                const float inv_std = inv_std_all[f];

                float sum_g = 0.0f, sum_g_xhat = 0.0f;
                for (int b = 0; b < batch; b++) {
                    const int idx = b * num_features + f;
                    const float gi = gamma_->data[f] * output->grad[idx];
                    sum_g      += gi;
                    sum_g_xhat += gi * x_hat[idx];
                }

                const float mean_g      = sum_g / N;
                const float mean_g_xhat = sum_g_xhat / N;

                for (int b = 0; b < batch; b++) {
                    const int idx = b * num_features + f;
                    const float gi = gamma_->data[f] * output->grad[idx];

                    if (input->requires_grad)
                        input->grad[idx] +=
                            inv_std * (gi - mean_g - x_hat[idx] * mean_g_xhat);

                    if (gamma_->requires_grad)
                        gamma_->grad[f] += output->grad[idx] * x_hat[idx];
                    if (beta_->requires_grad)
                        beta_->grad[f] += output->grad[idx];
                }
            }
        };
    }

    return output;
}

vector<TensorPtr> BatchNorm::parameters() {
    return {gamma, beta};
}
