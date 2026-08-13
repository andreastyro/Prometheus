#pragma once
#include "ml/nn/module.hpp"

// MaxPool2D — reduces spatial size by taking the maximum value in each region.
//
// Slides a window of size kernel_size x kernel_size across the feature map
// and keeps only the maximum value per window. This:
//   - Makes the representation smaller (faster, less memory)
//   - Makes features invariant to small shifts — "I found an edge near here"
//     rather than "I found an edge at exactly pixel (12, 7)"
//
// Input shape:  [batch, channels, height, width]
// Output shape: [batch, channels, height/stride, width/stride]
//
// stride defaults to kernel_size (non-overlapping windows).
class MaxPool2D : public Module {
public:
    int kernel_size; // size of the pooling window, e.g. 2 halves the spatial dimensions
    int stride;      // step between windows (-1 means use kernel_size, i.e. no overlap)

    MaxPool2D(int kernel_size, int stride = -1);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override; // no learnable parameters
};
