#pragma once
#include "ml/nn/module.hpp"

class BatchNorm : public Module {
public:
    Tensor gamma;   // learned scale
    Tensor beta;    // learned shift
    float eps;      // small value to avoid division by zero
    bool training;

    // num_features: number of columns (features) in the input
    BatchNorm(int num_features, float eps = 1e-5f, bool training = true);

    Tensor forward(Tensor& input) override;
    std::vector<Tensor*> parameters() override;
};
