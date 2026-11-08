#include "ml/utils/grad_clip.hpp"
#include <cmath>
#include "ml/optim/optimizer.hpp"
#include "../ops_internal.hpp"

float clip_grad_norm(std::vector<TensorPtr>& params, float max_norm) {
    double total_sq = 0.0;
    for (auto& p : params) {
        if (!p->has_grad()) continue;
        if (p->device() == Device::CUDA)
            total_sq += cuda_ops::sum_squares(device_grad(p), p->numel());
        else
            for (float g : p->grad_values()) total_sq += (double)g * g;
    }
    float total_norm = (float)std::sqrt(total_sq);

    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (auto& p : params) {
            if (!p->has_grad()) continue;
            if (p->device() == Device::CUDA)
                cuda_ops::scale_inplace(const_cast<float*>(device_grad(p)), p->numel(), scale);
            else
                for (float& g : p->grad_values()) g *= scale;
        }
    }
    return total_norm;
}
