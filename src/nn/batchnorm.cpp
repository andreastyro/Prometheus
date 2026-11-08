#include "ml/nn/batchnorm.hpp"
#include "ml/tensor.hpp"
#include <cmath>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <string>   // to_string; MSVC does not get it transitively
#include <vector>

using namespace std;

BatchNorm::BatchNorm(int num_features, float eps, bool training, float momentum)
    : eps(eps), training(training), momentum(momentum),
      running_mean(num_features, 0.0f), running_var(num_features, 1.0f) {
    gamma = Tensor::ones({num_features});
    beta  = Tensor::zeros({num_features});
    gamma->set_requires_grad(true);
    beta->set_requires_grad(true);
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
    const int ndim = (int)input->shape().size();
    if (ndim < 2)
        throw runtime_error("BatchNorm: expected at least 2 dimensions "
                            "[batch, features, ...]");

    const int N = input->shape()[0];
    const int C = input->shape()[1];
    if (C != (int)running_mean.size())
        throw runtime_error("BatchNorm: input has " + to_string(C) +
                            " channels but the layer was built for " +
                            to_string(running_mean.size()));

    int S = 1;
    for (int d = 2; d < ndim; d++) S *= input->shape()[d];

    auto output = make_shared<Tensor>(input->shape());

    // saved for the backward pass
    vector<float> x_hat(input->num_el());
    vector<float> inv_std_all(C);

    const int count = N * S;

    // Every channel is independent -- its statistics, its running averages,
    // its slice of the output and its gamma/beta gradient -- so they run in
    // parallel. They ran one at a time on one core: 10x slower than PyTorch.
    #pragma omp parallel for schedule(static) if((int64_t)C * count >= 65536)
    for (int c = 0; c < C; c++) {
        float mean, var;

        if (training) {
            const float* X = input->data();
            double acc = 0.0;
            for (int n = 0; n < N; n++) {
                const float* row = X + ((int64_t)n * C + c) * S;
                for (int s = 0; s < S; s++) acc += row[s];
            }
            mean = (float)(acc / count);

            acc = 0.0;
            for (int n = 0; n < N; n++) {
                const float* row = X + ((int64_t)n * C + c) * S;
                for (int s = 0; s < S; s++) { const float d = row[s] - mean; acc += (double)d * d; }
            }
            var = (float)(acc / count);

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
                const float norm = (input->data()[idx] - mean) * inv_std;
                x_hat[idx] = norm;
                output->data()[idx] = gamma->data()[c] * norm + beta->data()[c];
            }
    }

    if (input->requires_grad() || gamma->requires_grad() || beta->requires_grad()) {
        auto node = make_node(output, {input, gamma, beta});
        node->name = "BatchNorm::forward";

        auto gamma_ = gamma;
        auto beta_  = beta;
        const bool was_training = training;

        // moved, not copied: x_hat is the size of the whole activation
        node->backward_fn = [input, gamma_, beta_, x_hat = std::move(x_hat),
                             inv_std_all = std::move(inv_std_all),
                             N, C, S, count, was_training](const TensorPtr& output) {
            // Pointers hoisted and every per-channel sum kept in a register.
            // The loop used to add into gamma->grad()[c] and beta->grad()[c]
            // on every element, re-checking requires_grad each time: a store
            // to memory the compiler could not keep in a register, which made
            // this backward 3.9 ms against PyTorch's 1 ms for the whole layer.
            const float* G  = output->grad();
            const float* XH = x_hat.data();
            const float* gm = gamma_->data();
            float* GX = input->requires_grad() ? input->grad() : nullptr;
            float* GG = gamma_->requires_grad() ? gamma_->grad() : nullptr;
            float* GB = beta_->requires_grad() ? beta_->grad() : nullptr;

            #pragma omp parallel for schedule(static) if((int64_t)C * count >= 65536)
            for (int c = 0; c < C; c++) {
                const float inv_std = inv_std_all[c];
                const float g_c = gm[c];

                // sum(dy) and sum(dy * x_hat) over the channel: they give the
                // beta and gamma gradients and the input gradient's two means
                float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
                for (int n = 0; n < N; n++) {
                    const int64_t base = ((int64_t)n * C + c) * S;
                    const float* gy = G + base;
                    const float* xh = XH + base;
                    for (int s = 0; s < S; s++) {
                        sum_dy      += gy[s];
                        sum_dy_xhat += gy[s] * xh[s];
                    }
                }
                if (GG) GG[c] += sum_dy_xhat;
                if (GB) GB[c] += sum_dy;
                if (!GX) continue;

                // dx = inv_std * (g*dy - mean(g*dy) - x_hat * mean(g*dy*x_hat))
                const float mean_g      = was_training ? g_c * sum_dy / count : 0.0f;
                const float mean_g_xhat = was_training ? g_c * sum_dy_xhat / count : 0.0f;
                for (int n = 0; n < N; n++) {
                    const int64_t base = ((int64_t)n * C + c) * S;
                    const float* gy = G + base;
                    const float* xh = XH + base;
                    float* gx = GX + base;
                    for (int s = 0; s < S; s++)
                        gx[s] += inv_std * (g_c * gy[s] - mean_g - xh[s] * mean_g_xhat);
                }
            }
        };
    }

    return output;
}

vector<TensorPtr> BatchNorm::parameters() {
    return {gamma, beta};
}
