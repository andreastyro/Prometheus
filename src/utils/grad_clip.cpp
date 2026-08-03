#include "ml/utils/grad_clip.hpp"
#include <cmath>

float clip_grad_norm(std::vector<TensorPtr>& params, float max_norm) {
    float total_sq = 0.0f;
    for (auto& p : params)
        for (float g : p->grad)
            total_sq += g * g;
    float total_norm = std::sqrt(total_sq);

    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (auto& p : params)
            for (float& g : p->grad)
                g *= scale;
    }
    return total_norm;
}
