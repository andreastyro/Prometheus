#include "ml/nn/groupnorm.hpp"
#include "ml/ops.hpp"
#include "ml/tensor.hpp"
#include <cmath>
#include <cstdint>
#include <utility>
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

    // every (image, group) normalises on its own; pointers hoisted so the
    // inner loops are plain arrays the compiler can vectorise
    const float* X  = input->data();
    float*       O  = output->data();
    float*       XH = x_hat.data();
    const float* gm = gamma->data();
    const float* bt = beta->data();
    #pragma omp parallel for schedule(static) if((int64_t)batch * channels * spatial >= 65536)
    for (int bg = 0; bg < batch * num_groups; bg++) {
        const int b = bg / num_groups, g = bg % num_groups;
        {
            int c_start = g * group_size;
            int c_end   = c_start + group_size;

            // a group's channels are contiguous: one run of count elements
            const int count = group_size * spatial;
            const int64_t base = ((int64_t)b * channels + c_start) * spatial;
            const float* x = X + base;
            double acc = 0.0;
            for (int i = 0; i < count; i++) acc += x[i];
            const float mean = (float)(acc / count);
            acc = 0.0;
            for (int i = 0; i < count; i++) { const float d = x[i] - mean; acc += (double)d * d; }
            const float var = (float)(acc / count);

            float inv_std = 1.0f / std::sqrt(var + eps);
            inv_std_all[b * num_groups + g] = inv_std;

            // normalize and apply per-channel gamma / beta
            for (int c = c_start; c < c_end; c++) {
                const int64_t off = ((int64_t)b * channels + c) * spatial;
                const float gc = gm[c], bc = bt[c];
                for (int s = 0; s < spatial; s++) {
                    const float norm = (X[off + s] - mean) * inv_std;
                    XH[off + s] = norm;
                    O[off + s]  = gc * norm + bc;
                }
            }
        }
    }

    if (input->requires_grad() || gamma->requires_grad() || beta->requires_grad()) {
        auto node = make_node(output, {input, gamma, beta});
        node->name = "GroupNorm::forward";
        autograd::attach_reference(*node, [batch, channels, spatial, groups = num_groups, eps = eps](const std::vector<TensorPtr>& in) -> TensorPtr {
            auto xg  = reshape(in[0], {(int64_t)batch, (int64_t)groups, (int64_t)(channels / groups) * spatial});
            auto m   = mean(xg, 2, true);
            auto d   = subtract(xg, m);
            auto var = mean(multiply(d, d), 2, true);
            auto xh  = reshape(divide(d, sqrt_op(add(eps, var))), {(int64_t)batch, (int64_t)channels, (int64_t)spatial});
            auto out = add(multiply(xh, reshape(in[1], {1, (int64_t)channels, 1})),
                           reshape(in[2], {1, (int64_t)channels, 1}));
            return reshape(out, in[0]->shape());
        });

        auto gamma_ = gamma;
        auto beta_  = beta;
        int  groups = num_groups;

        // moved, not copied: x_hat is the size of the whole activation
        node->backward_fn = [input, gamma_, beta_, x_hat = std::move(x_hat),
                             inv_std_all = std::move(inv_std_all),
                             batch, channels, spatial, group_size, groups](const TensorPtr& output) {
            const int N = group_size * spatial;

            // The input gradient is per (image, group) and runs in parallel.
            // gamma and beta sum over every image, so they are computed in a
            // second pass parallel over channels, where each thread owns its
            // slot -- accumulating them inside the first loop would race.
            const float* G  = output->grad();
            const float* XH = x_hat.data();
            const float* gm = gamma_->data();
            float* GX = input->requires_grad() ? input->grad() : nullptr;
            if (GX)
            #pragma omp parallel for schedule(static) if((int64_t)batch * channels * spatial >= 65536)
            for (int bg = 0; bg < batch * groups; bg++) {
                const int b = bg / groups, g = bg % groups;
                {
                    const int c_start = g * group_size;
                    const int c_end   = c_start + group_size;
                    const float inv_std = inv_std_all[b * groups + g];

                    // Sums over the whole normalisation group.
                    float sum_g = 0.0f, sum_g_xhat = 0.0f;
                    for (int c = c_start; c < c_end; c++) {
                        const int64_t off = ((int64_t)b * channels + c) * spatial;
                        const float gc = gm[c];
                        for (int s = 0; s < spatial; s++) {
                            const float gi = gc * G[off + s];
                            sum_g      += gi;
                            sum_g_xhat += gi * XH[off + s];
                        }
                    }

                    const float mean_g      = sum_g / N;
                    const float mean_g_xhat = sum_g_xhat / N;

                    for (int c = c_start; c < c_end; c++) {
                        const int64_t off = ((int64_t)b * channels + c) * spatial;
                        const float gc = gm[c];
                        for (int s = 0; s < spatial; s++)
                            GX[off + s] += inv_std * (gc * G[off + s] - mean_g - XH[off + s] * mean_g_xhat);
                    }
                }
            }

            if (gamma_->requires_grad() || beta_->requires_grad()) {
                const float* G = output->grad();
                #pragma omp parallel for schedule(static) if((int64_t)batch * channels * spatial >= 65536)
                for (int c = 0; c < channels; c++) {
                    double gg = 0.0, gb = 0.0;
                    for (int b = 0; b < batch; b++) {
                        const int64_t base = ((int64_t)b * channels + c) * spatial;
                        for (int s = 0; s < spatial; s++) {
                            gg += (double)G[base + s] * x_hat[base + s];
                            gb += G[base + s];
                        }
                    }
                    if (gamma_->requires_grad()) gamma_->grad()[c] += (float)gg;
                    if (beta_->requires_grad())  beta_->grad()[c]  += (float)gb;
                }
            }
        };
    }

    return output;
}

vector<TensorPtr> GroupNorm::parameters() {
    return {gamma, beta};
}
