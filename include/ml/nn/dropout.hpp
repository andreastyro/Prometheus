#pragma once
#include "ml/nn/module.hpp"

class Dropout : public Module {
public:
    float rate;
    bool training;

    Dropout(float rate, bool training = true);

    Tensor forward(Tensor& input) override;
    std::vector<Tensor*> parameters() override;
};
