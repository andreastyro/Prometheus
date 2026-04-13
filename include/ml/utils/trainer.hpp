#pragma once
#include "ml/tensor.hpp"
#include "ml/nn/module.hpp"
#include "ml/data/dataloader.hpp"
#include "ml/optim/optimizer.hpp"
#include <functional>
#include <vector>

struct TrainHistory {
    std::vector<float> loss;
    std::vector<float> accuracy;
};

TrainHistory train(
    Module& model,
    DataLoader& loader,
    Optimizer& optimizer,
    std::function<TensorPtr(TensorPtr, TensorPtr)> loss_fn,
    int epochs,
    bool verbose = true
);
