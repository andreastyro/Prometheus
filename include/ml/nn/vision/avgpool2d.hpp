#pragma once
#include "ml/nn/module.hpp"

// AvgPool2D — reduces spatial size by averaging values in each region.
//
// Like MaxPool2D but uses the average instead of the maximum.
// MaxPool is more common in classification (preserves the strongest signal),
// while AvgPool is useful when you want a smoother downsampling.
//
// Input shape:  [batch, channels, height, width]
// Output shape: [batch, channels, height/stride, width/stride]
//
// stride defaults to kernel_size (non-overlapping windows).
class AvgPool2D : public Module {
public:
    int kernel_size; // size of the averaging window
    int stride;      // step between windows (-1 means use kernel_size, i.e. no overlap)

    AvgPool2D(int kernel_size, int stride = -1);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override; // no learnable parameters
};
