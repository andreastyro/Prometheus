#pragma once
#include "ml/tensor.hpp"
#include "ml/nn/module.hpp"
#include <vector>

class Sequential : public Module{
public:

    std::vector<Module*> layers;

    Sequential(std::vector<Module*> layers);

    Tensor forward(Tensor& input) override;
    std::vector<Tensor*> parameters() override;

};