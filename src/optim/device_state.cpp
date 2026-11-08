// Optimizer state on the GPU. See DeviceBuffers in ml/optim/optimizer.hpp.
#include "ml/optim/optimizer.hpp"
#include "ml/cuda.hpp"

float* DeviceBuffers::get(std::size_t i, const std::vector<float>& host) {
    if (dev.size() <= i) dev.resize(i + 1);
    if (!dev[i]) {
        dev[i] = std::make_shared<Storage>(host.size() * sizeof(float), DType::Float32, Device::CUDA, false);
        cuda::copy_to_device(dev[i]->data(), host.data(), host.size() * sizeof(float));
    }
    return static_cast<float*>(dev[i]->data());
}

void DeviceBuffers::to_host(std::size_t i, std::vector<float>& host) const {
    if (i < dev.size() && dev[i])
        cuda::copy_to_host(host.data(), dev[i]->data(), host.size() * sizeof(float));
}

float* device_weights(const TensorPtr& p) {
    if (p->dtype() != DType::Float32)
        throw std::runtime_error("optimizer: GPU parameters must be float32");
    return static_cast<float*>(p->storage()->data()) + p->offset();
}

const float* device_grad(const TensorPtr& p) {
    auto g = p->grad_tensor();
    return g ? static_cast<const float*>(g->storage()->data()) : nullptr;
}
