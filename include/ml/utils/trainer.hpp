#pragma once
#include "ml/tensor.hpp"
#include "ml/nn/module.hpp"
#include "ml/data/dataloader.hpp"
#include "ml/optim/optimizer.hpp"
#include <functional>

void train(
    Module& model,
    DataLoader& loader,
    Optimizer& optimizer,
    std::function<TensorPtr(TensorPtr, TensorPtr)> loss_fn,
    int epochs,
    bool verbose = true
);
