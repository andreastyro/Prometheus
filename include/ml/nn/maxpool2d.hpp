#pragma once
#include "ml/nn/module.hpp"

class MaxPool2D : public Module {
public:
    int kernel_size;
    int stride;

    MaxPool2D(int kernel_size, int stride = -1);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};
