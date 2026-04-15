#pragma once
#include "ml/nn/module.hpp"

class Flatten : public Module {
public:
    Flatten() = default;

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};
