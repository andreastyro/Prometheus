#pragma once
#include "ml/nn/module.hpp"

class AvgPool2D : public Module {
public:
    int kernel_size;
    int stride;

    AvgPool2D(int kernel_size, int stride = -1);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};
