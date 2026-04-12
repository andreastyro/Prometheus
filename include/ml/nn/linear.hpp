#pragma once
#include "ml/nn/module.hpp"

class Linear : public Module {
public:
    TensorPtr weights;
    TensorPtr bias;
    Linear(int in_features, int out_features);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};
