#include "ml/nn/groupnorm.hpp"
#include "ml/tensor.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace std;

GroupNorm::GroupNorm(int num_groups, int num_channels, float eps)
    : num_groups(num_groups), num_channels(num_channels), eps(eps) {

    if (num_channels % num_groups != 0)
        throw runtime_error("GroupNorm: num_channels must be divisible by num_groups");

    gamma = Tensor::ones({num_channels});
    beta  = Tensor::zeros({num_channels});
    gamma->set_requires_grad(true);
    beta->set_requires_grad(true);
}

// Backward for y = gamma * x_hat + beta, where x_hat = (x - mean) / std and the
// mean and variance are taken over one group of channels times the spatial
// extent, for one batch item. Let N be that group size and
//
//     g[j] = gamma[c(j)] * dL/dy[j]
//
// then, because mean and var both depend on every element of the group,
//
//     dL/dx[j] = (1/std) * ( g[j] - mean(g) - x_hat[j] * mean(g * x_hat) )
//
// Treating mean and var as constants here would drop the last two terms, which
// is the classic LayerNorm/GroupNorm backward mistake.
TensorPtr GroupNorm::forward(TensorPtr input) {
    // input: [batch, num_channels, spatial]
    // spatial = product of all dims after channels
    int batch    = input->shape()[0];
    int channels = input->shape()[1];
    int spatial  = input->num_el() / (batch * channels);

    if (channels != num_channels)
        throw runtime_error("GroupNorm: input channels do not match num_channels");

    int group_size = num_channels / num_groups;  // channels per group
    auto output = make_shared<Tensor>(input->shape());

    // saved for the backward pass
    vector<float> x_hat(input->num_el());
    vector<float> inv_std_all(batch * num_groups);

    for (int b = 0; b < batch; b++) {
        for (int g = 0; g < num_groups; g++) {
            int c_start = g * group_size;
            int c_end   = c_start + group_size;

            // compute mean over this group
            float mean = 0.0f;
            int count  = group_size * spatial;
            for (int c = c_start; c < c_end; c++)
                for (int s = 0; s < spatial; s++)
                    mean += input->data()[(b * channels + c) * spatial + s];
            mean /= count;

            // compute variance
            float var = 0.0f;
            for (int c = c_start; c < c_end; c++)
                for (int s = 0; s < spatial; s++) {
                    float diff = input->data()[(b * channels + c) * spatial + s] - mean;
                    var += diff * diff;
                }
            var /= count;

            float inv_std = 1.0f / std::sqrt(var + eps);
            inv_std_all[b * num_groups + g] = inv_std;

            // normalize and apply per-channel gamma / beta
            for (int c = c_start; c < c_end; c++)
                for (int s = 0; s < spatial; s++) {
                    int idx = (b * channels + c) * spatial + s;
                    float norm = (input->data()[idx] - mean) * inv_std;
                    x_hat[idx] = norm;
                    output->data()[idx] = gamma->data()[c] * norm + beta->data()[c];
                }
        }
    }

    if (input->requires_grad() || gamma->requires_grad() || beta->requires_grad()) {
        auto node = make_node(output, {input, gamma, beta});

        auto gamma_ = gamma;
        auto beta_  = beta;
        int  groups = num_groups;

        node->backward_fn = [input, output, gamma_, beta_, x_hat, inv_std_all,
                             batch, channels, spatial, group_size, groups]() {
            const int N = group_size * spatial;

            for (int b = 0; b < batch; b++) {
                for (int g = 0; g < groups; g++) {
                    const int c_start = g * group_size;
                    const int c_end   = c_start + group_size;
                    const float inv_std = inv_std_all[b * groups + g];

                    // Sums over the whole normalisation group.
                    float sum_g = 0.0f, sum_g_xhat = 0.0f;
                    for (int c = c_start; c < c_end; c++)
                        for (int s = 0; s < spatial; s++) {
                            const int idx = (b * channels + c) * spatial + s;
                            const float gi = gamma_->data()[c] * output->grad()[idx];
                            sum_g      += gi;
                            sum_g_xhat += gi * x_hat[idx];
                        }

                    const float mean_g      = sum_g / N;
                    const float mean_g_xhat = sum_g_xhat / N;

                    for (int c = c_start; c < c_end; c++)
                        for (int s = 0; s < spatial; s++) {
                            const int idx = (b * channels + c) * spatial + s;
                            const float gi = gamma_->data()[c] * output->grad()[idx];

                            if (input->requires_grad())
                                input->grad()[idx] +=
                                    inv_std * (gi - mean_g - x_hat[idx] * mean_g_xhat);

                            // gamma and beta are per channel, so accumulate
                            // across every batch item and spatial position.
                            if (gamma_->requires_grad())
                                gamma_->grad()[c] += output->grad()[idx] * x_hat[idx];
                            if (beta_->requires_grad())
                                beta_->grad()[c] += output->grad()[idx];
                        }
                }
            }
        };
    }

    return output;
}

vector<TensorPtr> GroupNorm::parameters() {
    return {gamma, beta};
}
