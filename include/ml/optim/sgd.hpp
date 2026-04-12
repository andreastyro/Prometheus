#pragma once
#include <vector>
#include "ml/tensor.hpp"
#include "ml/optim/optimizer.hpp"


class SGD : public Optimizer{
public:
    float lr;

    SGD(std::vector<Tensor*> params, float lr);

    void step() override;

};