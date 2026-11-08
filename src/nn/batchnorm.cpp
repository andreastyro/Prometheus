#include "ml/nn/batchnorm.hpp"
#include "ml/ops.hpp"
#include "ml/tensor.hpp"
#include "norm_kernels.hpp"
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
// Backward is the usual normalisation gradient, over N*S elements per channel:
//
//   dL/dx[i] = (1/std) * ( g[i] - mean(g) - x_hat[i] * mean(g * x_hat) )
//
// with g[i] = gamma[c] * dL/dy[i]. In eval mode the statistics come from the
// running estimate and are constants, so the last two terms vanish.
//
// Layout of the work. The tensor is N*C rows of S contiguous values. Each pass
// runs in parallel over rows, which keeps every core busy even when C is small,
// and each row is read while it sits in L1:
//
//   forward   row mean and M2 -> per-channel mean and variance (Chan's
//             combination, exact in double) -> y = a*(x - mean) + beta
//   backward  row sums of dy and dy*(x - mean) -> per-channel coefficients ->
//             dx = k1*dy + k2*(x - mean) + k3
//
// x_hat is not stored: the backward recomputes it from x and two numbers per
// channel. Storing it cost a full write and read of the activation.
TensorPtr BatchNorm::forward(TensorPtr input) {
    input = checked_input("BatchNorm", input);
    const int ndim = (int)input->shape().size();
    if (ndim < 2)
        throw runtime_error("BatchNorm: expected at least 2 dimensions "
                            "[batch, features, ...]");

    const int64_t N = input->shape()[0];
    const int64_t C = input->shape()[1];
    if (C != (int64_t)running_mean.size())
        throw runtime_error("BatchNorm: input has " + to_string(C) +
                            " channels but the layer was built for " +
                            to_string(running_mean.size()));

    int64_t S = 1;
    for (int d = 2; d < ndim; d++) S *= input->shape()[d];
    const int64_t rows  = N * C;
    const int64_t count = N * S;
    const bool    big   = rows * S >= 65536;

    auto output = Tensor::empty(input->shape());
    const float* X  = input->data();
    float*       Y  = output->data();
    const float* gm = gamma->data();
    const float* bt = beta->data();

    vector<float> mean_c(C), inv_std_c(C);
    if (training) {
        vector<double> rmean(rows), rm2(rows);
        #pragma omp parallel for schedule(static) if(big)
        for (int64_t r = 0; r < rows; r++)
            normk::moments(X + r * S, S, rmean[r], rm2[r]);

        for (int64_t c = 0; c < C; c++) {
            double m = 0.0;
            for (int64_t n = 0; n < N; n++) m += rmean[n * C + c];
            m /= (double)N;
            double m2 = 0.0;
            for (int64_t n = 0; n < N; n++) {
                const double d = rmean[n * C + c] - m;
                m2 += rm2[n * C + c] + (double)S * d * d;
            }
            const float var = count ? (float)(m2 / (double)count) : 0.0f;
            mean_c[c]    = (float)m;
            inv_std_c[c] = 1.0f / std::sqrt(var + eps);
            // exponential moving average, as PyTorch does: the running variance
            // takes the unbiased estimate (n / (n-1)), the normalisation the biased
            const float unbiased = count > 1 ? (float)(m2 / (double)(count - 1)) : var;
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * (float)m;
            running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * unbiased;
        }
    } else {
        for (int64_t c = 0; c < C; c++) {
            mean_c[c]    = running_mean[c];
            inv_std_c[c] = 1.0f / std::sqrt(running_var[c] + eps);
        }
    }

    #pragma omp parallel for schedule(static) if(big)
    for (int64_t r = 0; r < rows; r++) {
        const int64_t c = r % C;
        const float a = gm[c] * inv_std_c[c];
        normk::affine(X + r * S, Y + r * S, S, a, mean_c[c], bt[c]);
    }

    if (input->requires_grad() || gamma->requires_grad() || beta->requires_grad()) {
        auto node = make_node(output, {input, gamma, beta});
        node->name = "BatchNorm::forward";
        autograd::attach_reference(*node, [N, C, S, eps = eps, was_training = training, rm = running_mean, rv = running_var](const std::vector<TensorPtr>& in) -> TensorPtr {
            auto x3 = reshape(in[0], {N, C, S});
            TensorPtr xh;
            if (was_training) {
                auto m   = mean(mean(x3, 2, true), 0, true);
                auto d   = subtract(x3, m);
                auto var = mean(mean(multiply(d, d), 2, true), 0, true);
                xh = divide(d, sqrt_op(add(eps, var)));
            } else {
                // the running statistics are constants. They are copied as they stood
                // at the forward pass; the forward in eval mode does not update them.
                vector<float> mv(C), iv(C);
                for (int64_t c = 0; c < C; c++) { mv[c] = rm[c]; iv[c] = 1.0f / std::sqrt(rv[c] + eps); }
                xh = multiply(subtract(x3, make_shared<Tensor>(Shape{1, C, 1}, mv)),
                              make_shared<Tensor>(Shape{1, C, 1}, iv));
            }
            auto out = add(multiply(xh, reshape(in[1], {1, C, 1})), reshape(in[2], {1, C, 1}));
            return reshape(out, in[0]->shape());
        });

        auto gamma_ = gamma;
        auto beta_  = beta;
        const bool was_training = training;

        node->backward_fn = [input, gamma_, beta_, mean_c = std::move(mean_c),
                             inv_std_c = std::move(inv_std_c),
                             N, C, S, count, big, was_training](const TensorPtr& output) {
            const int64_t rows = N * C;
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
                normk::grad_sums(G + r * S, X + r * S, S, mean_c[r % C], rsg[r], rsgx[r]);

            // per channel: the gamma/beta gradients and dx's coefficients
            vector<float> k1(C), k2(C), k3(C);
            for (int64_t c = 0; c < C; c++) {
                double sum_dy = 0.0, sum_dyxc = 0.0;
                for (int64_t n = 0; n < N; n++) { sum_dy += rsg[n * C + c]; sum_dyxc += rsgx[n * C + c]; }
                const double inv_std     = inv_std_c[c];
                const double sum_dy_xhat = inv_std * sum_dyxc;
                if (GG) GG[c] += (float)sum_dy_xhat;
                if (GB) GB[c] += (float)sum_dy;

                // dx = inv_std * (g*dy - mean(g*dy) - x_hat * mean(g*dy*x_hat))
                const double mean_g      = was_training ? gm[c] * sum_dy / count : 0.0;
                const double mean_g_xhat = was_training ? gm[c] * sum_dy_xhat / count : 0.0;
                k1[c] = (float)(inv_std * gm[c]);
                k2[c] = (float)(-inv_std * inv_std * mean_g_xhat);
                k3[c] = (float)(-inv_std * mean_g);
            }
            if (!GX) return;

            #pragma omp parallel for schedule(static) if(big)
            for (int64_t r = 0; r < rows; r++) {
                const int64_t c = r % C;
                normk::affine_grad(G + r * S, X + r * S, GX + r * S, S,
                                   k1[c], k2[c], mean_c[c], k3[c]);
            }
        };
    }

    return output;
}

vector<TensorPtr> BatchNorm::parameters() {
    return {gamma, beta};
}
