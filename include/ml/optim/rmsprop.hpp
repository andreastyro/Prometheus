#pragma once
#include "ml/tensor.hpp"
#include "ml/optim/optimizer.hpp"
#include <vector>

class RMSprop : public Optimizer{
public:
    float lr;
    float beta;
    float eps;
    std::vector<std::vector<float>> v;

    RMSprop(std::vector<TensorPtr>, float lr = 0.001f, float beta = 0.9f, float eps = 1e-8f);

    void step() override;

};