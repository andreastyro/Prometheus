#pragma once
#include <vector>
#include "ml/tensor.hpp"
#include "ml/optim/optimizer.hpp"

class SGD : public Optimizer {
public:
    float lr;
    float momentum;
    std::vector<std::vector<float>> velocity;

    SGD(std::vector<TensorPtr> params, float lr, float momentum = 0.0f);

    void step() override;
};
