#pragma once
#include "ml/nn/module.hpp"

class Linear : public Module {
public:
    Tensor weights;
    Tensor bias;
    Linear(int in_features, int out_features);

    Tensor forward(Tensor& input) override;
    std::vector<Tensor*> parameters() override;
};
