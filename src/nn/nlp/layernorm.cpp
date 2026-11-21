#include "../../ops_internal.hpp"
#include "ml/nn/nlp/layernorm.hpp"
#include "ml/ops.hpp"
#include "ml/autograd.hpp"
#include <cmath>
#include <utility>

using namespace std;

LayerNorm::LayerNorm(int normalized_shape, float eps)
    : normalized_shape(normalized_shape), eps(eps) {
    gamma = Tensor::ones({normalized_shape});
    beta  = Tensor::zeros({normalized_shape});
    gamma->set_requires_grad(true);
    beta->set_requires_grad(true);
}

// LayerNorm backward (each row is independent):
//   y[i] = gamma[i] * x_hat[i] + beta[i],   x_hat[i] = (x[i] - mean) / std
//
//   Let g[i] = dy/dx_hat[i] * d_loss/dy[i] = gamma[i] * output.grad[i]
//   d_loss/d_x[j] = (1/std) * (1/N) * (N*g[j] - Σg - x_hat[j]*Σ(g*x_hat))
// The forward written with ops: the GPU path, and the reference the
// higher-order gradients differentiate.
static TensorPtr layernorm_ops(const TensorPtr& x, const TensorPtr& g, const TensorPtr& b, float eps) {
    const int last = x->dim() - 1;
    auto m   = mean(x, last, true);
    auto d   = subtract(x, m);
    auto var = mean(multiply(d, d), last, true);
    auto xh  = divide(d, sqrt_op(add(eps, var)));
    return add(multiply(xh, g), b);
}

TensorPtr LayerNorm::forward(TensorPtr input) { return forward_as(input, DType::Float32); }

TensorPtr LayerNorm::forward_as(TensorPtr input, DType out_dtype) {
    if (input->device() == Device::CUDA || gamma->device() == Device::CUDA) {
        // one fused kernel forward; the gradient differentiates the op form
        auto out = cuda_ops::layernorm(input, gamma, beta, eps, out_dtype);
        if (input->requires_grad() || gamma->requires_grad() || beta->requires_grad()) {
            auto node = make_node(out, {input, gamma, beta});
            node->name = "LayerNorm::forward";
            autograd::attach_reference(*node, [eps = eps](const std::vector<TensorPtr>& in) -> TensorPtr {
                return layernorm_ops(in[0], in[1], in[2], eps);
            });
            // a plain backward runs the fused kernel; a recorded one the op form
            auto reference = node->backward_graph;
            auto x = input, g = gamma, b = beta;
            const float e = eps;
            node->backward_graph = [reference, x, g, b, e](const TensorPtr& out, const TensorPtr& G) -> std::vector<TensorPtr> {
                if (grad_enabled()) return reference(out, G);
                // x's gradient buffer, when another branch (the residual stream)
                // already gave it one: the kernel adds into it directly
                auto r = cuda_ops::layernorm_backward(x, g, G, e,
                                                      backward_into_grads() && x->requires_grad() && x->has_grad()
                                                          ? x->grad_tensor() : nullptr);
                return {x->requires_grad() ? r[0] : nullptr, g->requires_grad() ? r[1] : nullptr,
                        b->requires_grad() ? r[2] : nullptr};
            };
        }
        return out;
    }
    input = checked_input("LayerNorm", input);
    int N       = normalized_shape;
    int num_rows = input->num_el() / N;
    // every element written below: no zero-fill; pointers hoisted so the
    // loops vectorise (per-element data() / grad() calls kept them scalar:
    // 4.9 ms against PyTorch's 1.2 for 2048 rows of 768, forward + backward)
    auto output  = make_shared<Tensor>(input->shape(), Tensor::Uninitialized{});
    const size_t n = (size_t)input->num_el();
    // x_hat and inv_std saved per row for the backward, uninitialised
    std::shared_ptr<float> x_hat_all(new float[n], std::default_delete<float[]>());
    std::shared_ptr<float> inv_std_all(new float[num_rows > 0 ? num_rows : 1], std::default_delete<float[]>());
    {
        const float* X = input->data();
        const float* G = gamma->data();
        const float* Bt = beta->data();
        float* Y = output->data();
        float* XH = x_hat_all.get();
        float* IS = inv_std_all.get();
        #pragma omp parallel for schedule(static) if(num_rows >= 64)
        for (int r = 0; r < num_rows; r++) {
            const float* x = X + (size_t)r * N;
            float mean_v = 0.0f;
            for (int i = 0; i < N; i++) mean_v += x[i];
            mean_v /= N;
            float var = 0.0f;
            for (int i = 0; i < N; i++) { const float d = x[i] - mean_v; var += d * d; }
            var /= N;
            const float is = 1.0f / sqrtf(var + eps);
            IS[r] = is;
            float* xh = XH + (size_t)r * N;
            float* y = Y + (size_t)r * N;
            for (int i = 0; i < N; i++) {
                xh[i] = (x[i] - mean_v) * is;
                y[i] = G[i] * xh[i] + Bt[i];
            }
        }
    }

    if (input->requires_grad() || gamma->requires_grad() || beta->requires_grad()) {
        auto g = gamma;
        auto b = beta;
        auto node = make_node(output, {input, g, b});
        node->name = "LayerNorm::forward";
        autograd::attach_reference(*node, [eps = eps](const std::vector<TensorPtr>& in) -> TensorPtr {
            return layernorm_ops(in[0], in[1], in[2], eps);
        });
        node->backward_fn = [input, g, b, x_hat_all, inv_std_all, num_rows, N](const TensorPtr& output) {
            const float* GO = output->grad();
            const float* GM = g->data();
            const float* XH = x_hat_all.get();
            const float* IS = inv_std_all.get();
            float* GX = input->requires_grad() ? input->grad() : nullptr;
            const bool need_gb = g->requires_grad() || b->requires_grad();
            float* GG = g->requires_grad() ? g->grad() : nullptr;
            float* GB = b->requires_grad() ? b->grad() : nullptr;
            #pragma omp parallel if(num_rows >= 64)
            {
                // gamma / beta: each thread sums its rows into its own pair of
                // rows (contiguous reads), added once at the end -- the old
                // column-at-a-time walk strode N floats per element
                std::vector<float> pg(need_gb ? (size_t)N : 0, 0.0f), pb(need_gb ? (size_t)N : 0, 0.0f);
                #pragma omp for schedule(static) nowait
                for (int r = 0; r < num_rows; r++) {
                    const float* go = GO + (size_t)r * N;
                    const float* xh = XH + (size_t)r * N;
                    if (need_gb)
                        for (int i = 0; i < N; i++) { pg[i] += go[i] * xh[i]; pb[i] += go[i]; }
                    if (GX) {
                        float sum_g = 0.0f, sum_g_xhat = 0.0f;
                        for (int i = 0; i < N; i++) {
                            const float gi = GM[i] * go[i];
                            sum_g += gi;
                            sum_g_xhat += gi * xh[i];
                        }
                        const float is = IS[r], k = is / N;
                        float* gx = GX + (size_t)r * N;
                        for (int i = 0; i < N; i++)
                            gx[i] += k * (N * GM[i] * go[i] - sum_g - xh[i] * sum_g_xhat);
                    }
                }
                if (need_gb) {
                    #pragma omp critical
                    for (int i = 0; i < N; i++) {
                        if (GG) GG[i] += pg[i];
                        if (GB) GB[i] += pb[i];
                    }
                }
            }
        };
    }

    return output;
}

vector<TensorPtr> LayerNorm::parameters() {
    return {gamma, beta};
}
