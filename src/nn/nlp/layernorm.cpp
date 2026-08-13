#include "ml/nn/nlp/layernorm.hpp"
#include "ml/autograd.hpp"
#include <cmath>

using namespace std;

LayerNorm::LayerNorm(int normalized_shape, float eps)
    : normalized_shape(normalized_shape), eps(eps) {
    gamma = Tensor::ones({normalized_shape});
    beta  = Tensor::zeros({normalized_shape});
    gamma->requires_grad = true;
    beta->requires_grad  = true;
}

// LayerNorm backward (each row is independent):
//   y[i] = gamma[i] * x_hat[i] + beta[i],   x_hat[i] = (x[i] - mean) / std
//
//   Let g[i] = dy/dx_hat[i] * d_loss/dy[i] = gamma[i] * output.grad[i]
//   d_loss/d_x[j] = (1/std) * (1/N) * (N*g[j] - Σg - x_hat[j]*Σ(g*x_hat))
TensorPtr LayerNorm::forward(TensorPtr input) {
    int N       = normalized_shape;
    int num_rows = input->num_el() / N;
    auto output  = make_shared<Tensor>(input->shape);

    // x_hat and inv_std saved per-row for the backward
    vector<float> x_hat_all(input->num_el());
    vector<float> inv_std_all(num_rows);

    for (int r = 0; r < num_rows; r++) {
        int off = r * N;

        float mean_v = 0.0f;
        for (int i = 0; i < N; i++) mean_v += input->data[off + i];
        mean_v /= N;

        float var = 0.0f;
        for (int i = 0; i < N; i++) {
            float d = input->data[off + i] - mean_v;
            var += d * d;
        }
        var /= N;

        float is = 1.0f / sqrtf(var + eps);
        inv_std_all[r] = is;

        for (int i = 0; i < N; i++) {
            x_hat_all[off + i] = (input->data[off + i] - mean_v) * is;
            output->data[off + i] = gamma->data[i] * x_hat_all[off + i] + beta->data[i];
        }
    }

    if (input->requires_grad || gamma->requires_grad || beta->requires_grad) {
        auto g = gamma;
        auto b = beta;
        auto node = make_node(output, {input, g, b});
        node->backward_fn = [input, g, b, output, x_hat_all, inv_std_all, num_rows, N]() {
            for (int r = 0; r < num_rows; r++) {
                int off = r * N;
                float is = inv_std_all[r];

                // g[i] = gamma[i] * output.grad[r, i]  (grad w.r.t. x_hat)
                float sum_g = 0.0f, sum_g_xhat = 0.0f;
                for (int i = 0; i < N; i++) {
                    float gi = g->data[i] * output->grad[off + i];
                    sum_g      += gi;
                    sum_g_xhat += gi * x_hat_all[off + i];
                }

                if (input->requires_grad) {
                    for (int i = 0; i < N; i++) {
                        float gi = g->data[i] * output->grad[off + i];
                        input->grad[off + i] += (is / N) * (N * gi - sum_g - x_hat_all[off + i] * sum_g_xhat);
                    }
                }
            }

            // gamma and beta accumulate over all rows
            if (g->requires_grad) {
                for (int r = 0; r < num_rows; r++) {
                    int off = r * N;
                    for (int i = 0; i < N; i++)
                        g->grad[i] += output->grad[off + i] * x_hat_all[off + i];
                }
            }
            if (b->requires_grad) {
                for (int r = 0; r < num_rows; r++) {
                    int off = r * N;
                    for (int i = 0; i < N; i++)
                        b->grad[i] += output->grad[off + i];
                }
            }
        };
    }

    return output;
}

vector<TensorPtr> LayerNorm::parameters() {
    return {gamma, beta};
}
