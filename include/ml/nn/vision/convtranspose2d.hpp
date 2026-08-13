#pragma once
#include "ml/nn/module.hpp"

// ConvTranspose2D — "deconvolution" / upsampling layer.
//
// The reverse of Conv2D: instead of shrinking the spatial dimensions,
// it expands them. Used in:
//   - Generative models (VAEs, GANs) to decode a small latent vector into an image
//   - Segmentation networks (U-Net decoder) to restore full-resolution predictions
//   - Any architecture where you need to go from small feature maps back to large ones
//
// Input shape:  [batch, in_channels, height, width]
// Output shape: [batch, out_channels, out_h, out_w]
//   where out_h = (height - 1) * stride - 2*padding + kernel_size
class ConvTranspose2D : public Module {
public:
    TensorPtr weights; // [in_channels, out_channels, kernel_size, kernel_size]
    TensorPtr bias;    // [out_channels]

    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;

    ConvTranspose2D(int in_channels, int out_channels, int kernel_size,
                    int stride = 1, int padding = 0, std::string weight_init = "default");

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};
