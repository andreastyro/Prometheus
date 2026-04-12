#pragma once
#include "ml/nn/module.hpp"

class ReLU : public Module {
public:
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};

class Sigmoid : public Module {
public:
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};

class Tanh : public Module {
public:
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};

class Softmax : public Module {
public:
    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};
