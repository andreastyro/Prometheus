#pragma once
#include "ml/tensor.hpp"
#include <vector>

class Module {
public:

    // Every layer must implement forward — takes input, returns output
    virtual Tensor forward(Tensor& input) = 0;

    // Every layer must return its learnable parameters (weights, biases)
    // Returns pointers so the optimizer can modify them directly
    virtual std::vector<Tensor*> parameters() = 0;

    // Virtual destructor — required for safe inheritance
    virtual ~Module() {}
};
