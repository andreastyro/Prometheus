#include "ml/optim/grad_scaler.hpp"

using namespace std;

GradScaler::GradScaler(float init_scale, float growth_factor,
                       float backoff_factor, int growth_interval, bool enabled)
    : scale(init_scale), growth_factor(growth_factor),
      backoff_factor(backoff_factor), growth_interval(growth_interval),
      step_count(0), enabled(enabled) {}

TensorPtr GradScaler::scale_loss(TensorPtr loss) {
    if (!enabled) return loss;
    auto out = make_shared<Tensor>(loss->shape);
    for (int i = 0; i < (int)loss->data.size(); i++)
        out->data[i] = loss->data[i] * scale;
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
