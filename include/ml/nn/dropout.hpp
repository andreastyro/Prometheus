#pragma once
#include "ml/nn/module.hpp"

class Dropout : public Module {
public:
    float rate;
    bool training;

    Dropout(float rate, bool training = true);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};
