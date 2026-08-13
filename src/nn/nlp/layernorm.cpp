#include "ml/nn/nlp/layernorm.hpp"
#include <cmath>

using namespace std;

LayerNorm::LayerNorm(int normalized_shape, float eps)
    : normalized_shape(normalized_shape), eps(eps) {

    gamma = Tensor::ones({normalized_shape});
    beta  = Tensor::zeros({normalized_shape});
    gamma->requires_grad = true;
    beta->requires_grad  = true;
}

TensorPtr LayerNorm::forward(TensorPtr input) {
    // treat input as a flat list of rows, each of length normalized_shape
    int num_rows = input->num_el() / normalized_shape;
    auto output  = make_shared<Tensor>(input->shape);

    for (int r = 0; r < num_rows; r++) {
        int offset = r * normalized_shape;

        // compute mean
        float mean = 0.0f;
        for (int i = 0; i < normalized_shape; i++)
            mean += input->data[offset + i];
        mean /= normalized_shape;

        // compute variance
        float var = 0.0f;
        for (int i = 0; i < normalized_shape; i++) {
            float diff = input->data[offset + i] - mean;
            var += diff * diff;
        }
        var /= normalized_shape;

        float inv_std = 1.0f / std::sqrt(var + eps);

        // normalize then apply gamma and beta
        for (int i = 0; i < normalized_shape; i++) {
            float norm = (input->data[offset + i] - mean) * inv_std;
            output->data[offset + i] = gamma->data[i] * norm + beta->data[i];
        }
    }

    return output;
}

vector<TensorPtr> LayerNorm::parameters() {
    return {gamma, beta};
}
