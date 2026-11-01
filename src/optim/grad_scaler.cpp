#include "ml/tensor.hpp"
#include "ml/optim/grad_scaler.hpp"

using namespace std;

GradScaler::GradScaler(float init_scale, float growth_factor,
                       float backoff_factor, int growth_interval, bool enabled)
    : scale(init_scale), growth_factor(growth_factor),
      backoff_factor(backoff_factor), growth_interval(growth_interval),
      step_count(0), enabled(enabled) {}

// scaled = loss * scale, so d(scaled)/d(loss) = scale. Without this node the
// scaled loss is detached from the graph and backward() reaches nothing, which
// silently disables training for anyone using the mixed-precision path.
TensorPtr GradScaler::scale_loss(TensorPtr loss) {
    if (!enabled) return loss;
    auto out = make_shared<Tensor>(loss->shape);
    for (int i = 0; i < (int)loss->data.size(); i++)
        out->data[i] = loss->data[i] * scale;

    if (loss->requires_grad) {
        float s = scale;
        auto node = make_node(out, {loss});
        node->backward_fn = [loss, out, s]() {
            for (int i = 0; i < out->num_el(); i++)
                loss->grad[i] += out->grad[i] * s;
        };
    }

    return out;
}

bool GradScaler::unscale(vector<TensorPtr> params) {
    if (!enabled) return true;
    bool clean = true;
    for (auto& p : params) {
        for (float& g : p->grad) {
            if (std::isinf(g) || std::isnan(g)) {
                clean = false;
                g = 0.0f;
            } else {
                g /= scale;
            }
        }
    }
    return clean;
}

void GradScaler::update(bool had_overflow) {
    if (!enabled) return;
    step_count++;
    if (had_overflow) {
        scale *= backoff_factor;
        step_count = 0;
    } else if (step_count % growth_interval == 0) {
        scale *= growth_factor;
    }
}
