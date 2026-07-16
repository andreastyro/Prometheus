#include "ml/nn/groupnorm.hpp"
#include <cmath>
#include <stdexcept>

using namespace std;

GroupNorm::GroupNorm(int num_groups, int num_channels, float eps)
    : num_groups(num_groups), num_channels(num_channels), eps(eps) {

    if (num_channels % num_groups != 0)
        throw runtime_error("GroupNorm: num_channels must be divisible by num_groups");

    gamma = Tensor::ones({num_channels});
    beta  = Tensor::zeros({num_channels});
    gamma->requires_grad = true;
    beta->requires_grad  = true;
}

TensorPtr GroupNorm::forward(TensorPtr input) {
    // input: [batch, num_channels, spatial]
    // spatial = product of all dims after channels
    int batch    = input->shape[0];
    int channels = input->shape[1];
    int spatial  = input->num_el() / (batch * channels);

    if (channels != num_channels)
        throw runtime_error("GroupNorm: input channels don't match num_channels");

    int group_size = num_channels / num_groups;  // channels per group
    auto output = make_shared<Tensor>(input->shape);

    for (int b = 0; b < batch; b++) {
        for (int g = 0; g < num_groups; g++) {
            int c_start = g * group_size;
            int c_end   = c_start + group_size;

            // compute mean over this group
            float mean = 0.0f;
            int count  = group_size * spatial;
            for (int c = c_start; c < c_end; c++)
                for (int s = 0; s < spatial; s++)
                    mean += input->data[(b * channels + c) * spatial + s];
            mean /= count;

            // compute variance
            float var = 0.0f;
            for (int c = c_start; c < c_end; c++)
                for (int s = 0; s < spatial; s++) {
                    float diff = input->data[(b * channels + c) * spatial + s] - mean;
                    var += diff * diff;
                }
            var /= count;

            float inv_std = 1.0f / std::sqrt(var + eps);

            // normalize and apply per-channel gamma / beta
            for (int c = c_start; c < c_end; c++)
                for (int s = 0; s < spatial; s++) {
                    int idx = (b * channels + c) * spatial + s;
                    float norm = (input->data[idx] - mean) * inv_std;
                    output->data[idx] = gamma->data[c] * norm + beta->data[c];
                }
        }
    }

    return output;
}

vector<TensorPtr> GroupNorm::parameters() {
    return {gamma, beta};
}
