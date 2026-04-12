#pragma once
#include "ml/tensor.hpp"
#include "ml/nn/module.hpp"
#include <vector>

class Sequential : public Module {
public:

    std::vector<Module*> layers;

    Sequential(std::vector<Module*> layers);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};
