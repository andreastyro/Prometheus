#pragma once
#include <vector>
#include "ml/tensor.hpp"

class Optimizer {
public:

    std::vector<Tensor*> parameters;

    virtual void step() = 0;

    void zero_grad() {
        for (Tensor* p : parameters)
            p->reset_grad();
    }

};