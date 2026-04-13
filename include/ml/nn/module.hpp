#pragma once
#include "ml/tensor.hpp"
#include <vector>

enum class WeightInit { DEFAULT, XAVIER, KAIMING };

class Module {
public:

    // Every layer must implement forward — takes input, returns output
    virtual TensorPtr forward(TensorPtr input) = 0;

    // Every layer must return its learnable parameters (weights, biases)
    virtual std::vector<TensorPtr> parameters() = 0;

    // Virtual destructor — required for safe inheritance
    virtual ~Module() {}
};
