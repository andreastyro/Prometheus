#pragma once
#include "ml/nn/module.hpp"

class ConvTranspose2D : public Module {
public:
    TensorPtr weights;
    TensorPtr bias;
    int in_channels, out_channels, kernel_size, stride, padding;

    ConvTranspose2D(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, std::string weight_init = "default");

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};
