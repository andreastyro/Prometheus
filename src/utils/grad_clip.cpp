#include "ml/utils/grad_clip.hpp"
#include <cmath>
#include "ml/optim/optimizer.hpp"
#include "../ops_internal.hpp"

float clip_grad_norm(std::vector<TensorPtr>& params, float max_norm, bool return_norm) {
    // All on the GPU: one launch for the sum of squares over every gradient,
    // one that scales them reading the norm on the device, and a single read
    // of the norm to return. It was a read back per parameter (52 stalls of
    // the stream in a GPT step) before scaling could start.
    std::vector<cuda_ops::OptTensor> gpu;
    bool cpu = false;
    for (auto& p : params) {
        if (!p->has_grad()) continue;
        if (p->device() == Device::CUDA)
            gpu.push_back({nullptr, device_grad(p), nullptr, nullptr, (long long)p->numel()});
        else
            cpu = true;
    }
    if (!gpu.empty() && !cpu) return cuda_ops::clip_grad_norm_multi(gpu, max_norm, nullptr, return_norm);

    double total_sq = 0.0;
    // parameters on both devices: the GPU's share first (no scaling), then all
    if (!gpu.empty()) cuda_ops::clip_grad_norm_multi(gpu, -1.0f, &total_sq);
    for (auto& p : params) {
        if (!p->has_grad()) continue;
        if (p->device() != Device::CUDA)
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
