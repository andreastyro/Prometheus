#pragma once
#include <vector>
#include "ml/tensor.hpp"
#include "ml/optim/optimizer.hpp"

class Adam : public Optimizer {
public:
    float lr;
    float beta1;
    float beta2;
    float eps;
    int t;  // step counter

    std::vector<std::vector<float>> m;  // first moment
    std::vector<std::vector<float>> v;  // second moment

    Adam(std::vector<Tensor*> params, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

    void step() override;
};
