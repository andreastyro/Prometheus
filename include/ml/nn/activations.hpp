#pragma once
#include "ml/nn/module.hpp"

class ReLU : public Module {
public:
    Tensor forward(Tensor& input) override;
    std::vector<Tensor*> parameters() override;
};

class Sigmoid : public Module {
public:
    Tensor forward(Tensor &input) override;
    std::vector<Tensor*> parameters() override;
};

class Tanh : public Module {
public:
    Tensor forward(Tensor& input) override;
    std::vector<Tensor*> parameters() override;
};

class Softmax : public Module {
public:
    Tensor forward(Tensor& input) override;
    std::vector<Tensor*> parameters() override;
};
