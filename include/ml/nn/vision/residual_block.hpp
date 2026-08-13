#pragma once
#include "ml/nn/module.hpp"
#include "ml/nn/vision/conv2d.hpp"
#include "ml/nn/groupnorm.hpp"
#include <memory>

// ResidualBlock — the building block of ResNet-style architectures.
//
// Forward path:  Conv3x3 → GroupNorm → ReLU → Conv3x3 → GroupNorm
// Skip path:     identity  (or 1×1 Conv if channels/stride don't match)
// Output:        add(forward_path, skip_path) → ReLU
//
// The projection conv is created automatically when in_channels != out_channels
// or stride != 1, so the skip tensor always matches the main path's shape before
// the residual add.
//
// num_groups controls GroupNorm — must evenly divide out_channels.
// A common default is 1 (InstanceNorm-like) for small channel counts,
// or 32 for larger channel counts typical in ResNet-50+.
//
// Input:  [batch, in_channels, H, W]
// Output: [batch, out_channels, H/stride, W/stride]
class ResidualBlock : public Module {
public:
    Conv2D    conv1;
    GroupNorm norm1;
    Conv2D    conv2;
    GroupNorm norm2;

    std::unique_ptr<Conv2D> proj; // 1x1 projection when shapes differ; null otherwise

    // in_channels:  channels coming in
    // out_channels: channels going out (and used in both conv layers)
    // stride:       applied to conv1 and the projection — use 2 to halve spatial dims
    // num_groups:   GroupNorm groups; must divide out_channels evenly
    ResidualBlock(int in_channels, int out_channels, int stride = 1, int num_groups = 1);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};
