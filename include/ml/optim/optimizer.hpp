#pragma once
#include <vector>
#include "ml/tensor.hpp"

class Optimizer {
public:

    std::vector<TensorPtr> parameters;

    virtual void step() = 0;

    void zero_grad() {
        for (auto& p : parameters)
            p->reset_grad();
    }
};
