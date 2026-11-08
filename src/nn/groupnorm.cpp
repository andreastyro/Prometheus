#include "ml/nn/groupnorm.hpp"
#include "ml/ops.hpp"
#include "ml/tensor.hpp"
#include "norm_kernels.hpp"
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
//
// The work runs over rows -- one (image, channel) pair, S contiguous values --
// in parallel, so a layer with few groups still uses every core. Row moments
// combine into group statistics exactly (Chan's formula, in double). The
// backward's row sums of dy and dy*(x - mean) give both the gamma/beta
// gradients and the group means of the formula above, so the activation is
// read twice in the backward and x_hat is never stored.
TensorPtr GroupNorm::forward(TensorPtr input) {
    input = checked_input("GroupNorm", input);
    if (input->dim() < 2)
        throw runtime_error("GroupNorm: expected [batch, channels, ...]");
    const int64_t batch    = input->shape()[0];
    const int64_t channels = input->shape()[1];
    if (channels != num_channels)
        throw runtime_error("GroupNorm: input channels do not match num_channels");
    const int64_t spatial = batch * channels ? input->num_el() / (batch * channels) : 0;

    const int64_t groups     = num_groups;
    const int64_t group_size = channels / groups;   // channels per group
    const int64_t rows       = batch * channels;
    const int64_t count      = group_size * spatial;
    const bool    big        = rows * spatial >= 65536;

    auto output = Tensor::empty(input->shape());
    const float* X  = input->data();
    float*       Y  = output->data();
    const float* gm = gamma->data();
    const float* bt = beta->data();

    vector<double> rmean(rows), rm2(rows);
    #pragma omp parallel for schedule(static) if(big)
    for (int64_t r = 0; r < rows; r++)
        normk::moments(X + r * spatial, spatial, rmean[r], rm2[r]);

    // group statistics: row r = b * channels + c, group rows are consecutive
    vector<float> mean_bg(batch * groups), inv_std_bg(batch * groups);
    for (int64_t bg = 0; bg < batch * groups; bg++) {
        const int64_t r0 = bg * group_size;
        double m = 0.0;
        for (int64_t k = 0; k < group_size; k++) m += rmean[r0 + k];
        m /= (double)group_size;
        double m2 = 0.0;
        for (int64_t k = 0; k < group_size; k++) {
            const double d = rmean[r0 + k] - m;
            m2 += rm2[r0 + k] + (double)spatial * d * d;
        }
        const float var = count ? (float)(m2 / (double)count) : 0.0f;
        mean_bg[bg]    = (float)m;
        inv_std_bg[bg] = 1.0f / std::sqrt(var + eps);
    }

    #pragma omp parallel for schedule(static) if(big)
    for (int64_t r = 0; r < rows; r++) {
        const int64_t c = r % channels, bg = r / group_size;
        const float a = gm[c] * inv_std_bg[bg];
        normk::affine(X + r * spatial, Y + r * spatial, spatial, a, mean_bg[bg], bt[c]);
    }

    if (input->requires_grad() || gamma->requires_grad() || beta->requires_grad()) {
        auto node = make_node(output, {input, gamma, beta});
        node->name = "GroupNorm::forward";
        autograd::attach_reference(*node, [batch, channels, spatial, groups, eps = eps](const std::vector<TensorPtr>& in) -> TensorPtr {
            auto xg  = reshape(in[0], {batch, groups, (channels / groups) * spatial});
            auto m   = mean(xg, 2, true);
            auto d   = subtract(xg, m);
            auto var = mean(multiply(d, d), 2, true);
            auto xh  = reshape(divide(d, sqrt_op(add(eps, var))), {batch, channels, spatial});
            auto out = add(multiply(xh, reshape(in[1], {1, channels, 1})),
                           reshape(in[2], {1, channels, 1}));
            return reshape(out, in[0]->shape());
        });

        auto gamma_ = gamma;
        auto beta_  = beta;

        node->backward_fn = [input, gamma_, beta_, mean_bg = std::move(mean_bg),
                             inv_std_bg = std::move(inv_std_bg),
                             batch, channels, spatial, groups, group_size, count, big](const TensorPtr& output) {
            const int64_t rows = batch * channels;
            const float* G  = output->grad();
            const float* X  = input->data();
            const float* gm = gamma_->data();
            float* GX = input->requires_grad() ? input->grad() : nullptr;
            float* GG = gamma_->requires_grad() ? gamma_->grad() : nullptr;
            float* GB = beta_->requires_grad() ? beta_->grad() : nullptr;

            // per row: sum(dy) and sum(dy * (x - mean))
            vector<double> rsg(rows), rsgx(rows);
            #pragma omp parallel for schedule(static) if(big)
            for (int64_t r = 0; r < rows; r++)
                normk::grad_sums(G + r * spatial, X + r * spatial, spatial,
                                 mean_bg[r / group_size], rsg[r], rsgx[r]);

            // gamma/beta: d gamma_c = sum_b inv_std * sum(dy*(x-mean)), d beta_c = sum_b sum(dy)
            for (int64_t c = 0; c < channels; c++) {
                double gg = 0.0, gb = 0.0;
                for (int64_t b = 0; b < batch; b++) {
                    const int64_t r = b * channels + c;
                    gg += inv_std_bg[r / group_size] * rsgx[r];
                    gb += rsg[r];
                }
                if (GG) GG[c] += (float)gg;
                if (GB) GB[c] += (float)gb;
            }
            if (!GX) return;

            // per (image, group): mean(g) and mean(g * x_hat), g = gamma * dy
            vector<float> k2(batch * groups), k3(batch * groups);
            for (int64_t bg = 0; bg < batch * groups; bg++) {
                const int64_t r0 = bg * group_size;
                const double inv_std = inv_std_bg[bg];
                double sg = 0.0, sgx = 0.0;
                for (int64_t k = 0; k < group_size; k++) {
                    const int64_t c = r0 % channels + k;
                    sg  += gm[c] * rsg[r0 + k];
                    sgx += gm[c] * rsgx[r0 + k];
                }
                const double mean_g      = count ? sg / count : 0.0;
                const double mean_g_xhat = count ? inv_std * sgx / count : 0.0;
                k2[bg] = (float)(-inv_std * inv_std * mean_g_xhat);
                k3[bg] = (float)(-inv_std * mean_g);
            }

            #pragma omp parallel for schedule(static) if(big)
            for (int64_t r = 0; r < rows; r++) {
                const int64_t c = r % channels, bg = r / group_size;
                normk::affine_grad(G + r * spatial, X + r * spatial, GX + r * spatial, spatial,
                                   gm[c] * inv_std_bg[bg], k2[bg], mean_bg[bg], k3[bg]);
            }
        };
    }

    return output;
}

vector<TensorPtr> GroupNorm::parameters() {
    return {gamma, beta};
}
