#include "ml/nn/batchnorm.hpp"
#include <cmath>

using namespace std;

BatchNorm::BatchNorm(int num_features, float eps, bool training)
    : eps(eps), training(training) {
    gamma = Tensor::ones({num_features});
    beta  = Tensor::zeros({num_features});
    gamma->requires_grad = true;
    beta->requires_grad  = true;
}

TensorPtr BatchNorm::forward(TensorPtr input) {
    // input: [batch, num_features]
    int batch        = input->shape[0];
    int num_features = input->shape[1];
    auto output      = make_shared<Tensor>(input->shape);

    for (int f = 0; f < num_features; f++) {
        // mean over batch for this feature
        float mean = 0.0f;
        for (int b = 0; b < batch; b++)
            mean += input->data[b * num_features + f];
        mean /= batch;

        // variance over batch
        float var = 0.0f;
        for (int b = 0; b < batch; b++) {
            float diff = input->data[b * num_features + f] - mean;
            var += diff * diff;
        }
        var /= batch;

        float inv_std = 1.0f / std::sqrt(var + eps);

        for (int b = 0; b < batch; b++) {
            float norm = (input->data[b * num_features + f] - mean) * inv_std;
            output->data[b * num_features + f] = gamma->data[f] * norm + beta->data[f];
        }
    }

    return output;
}

vector<TensorPtr> BatchNorm::parameters() {
    return {gamma, beta};
}
