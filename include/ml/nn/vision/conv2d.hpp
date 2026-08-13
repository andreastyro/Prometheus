#pragma once
#include "ml/nn/module.hpp"

/// 2D convolutional layer for image and spatial data.
///
/// Slides a small filter (kernel) across the input and computes a dot product
/// at each position. This detects local patterns (edges, textures, shapes)
/// wherever they appear in the image — the same filter is reused everywhere.
///
/// Input shape:  [batch, in_channels, height, width]
/// Output shape: [batch, out_channels, out_h, out_w]
///   where out_h = (height + 2*padding - kernel_size) / stride + 1
///
/// Example: Conv2D(1, 32, 3) — 1-channel greyscale input, 32 filters of size 3x3
class Conv2D : public Module {
public:
    TensorPtr weights; ///< [out_channels, in_channels, kernel_size, kernel_size]
    TensorPtr bias;    ///< [out_channels] — one bias per output filter

    int in_channels;  ///< Channels in the input (1=greyscale, 3=RGB)
    int out_channels; ///< Number of filters to learn (= number of output channels)
    int kernel_size;  ///< Width and height of each filter, e.g. 3 for a 3x3 filter
    int stride;       ///< Pixels to move the filter each step — 1 means every pixel
    int padding;      ///< Zero-padding added around the input border to control output size

    /// @param in_channels   channels in the input tensor
    /// @param out_channels  number of filters (output channels) to learn
    /// @param kernel_size   filter width and height (square filters only)
    /// @param stride        step size when sliding the filter (default 1)
    /// @param padding       border padding in pixels (default 0)
    /// @param weight_init   initialisation strategy: "default", "xavier", or "kaiming"
    Conv2D(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0, std::string weight_init = "default");

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override; ///< Returns {weights, bias}
};
