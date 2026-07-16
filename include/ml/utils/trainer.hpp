#pragma once
#include "ml/tensor.hpp"
#include "ml/nn/module.hpp"
#include "ml/data/dataloader.hpp"
#include "ml/optim/optimizer.hpp"
#include <functional>
#include <vector>

// TrainHistory — records loss and accuracy after each epoch.
// Useful for plotting learning curves or detecting overfitting.
struct TrainHistory {
    std::vector<float> loss;     // training loss per epoch
    std::vector<float> accuracy; // training accuracy per epoch (if labels are categorical)
};

// train() — runs a complete training loop.
//
// Each epoch:
//   1. Iterate through all mini-batches from the loader
//   2. Run forward pass
//   3. Compute loss using loss_fn
//   4. Backpropagate and update weights
//   5. Record loss and accuracy
//
// Args:
//   model:    the neural network to train
//   loader:   provides mini-batches of (x, y)
//   optimizer: updates weights after each backward pass
//   loss_fn:  a function(pred, target) -> TensorPtr, e.g. mse_loss
//   epochs:   number of full passes through the dataset
//   verbose:  if true, print loss and accuracy after each epoch
//
// Returns a TrainHistory with per-epoch loss and accuracy.
TrainHistory train(
    Module& model,
    DataLoader& loader,
    Optimizer& optimizer,
    std::function<TensorPtr(TensorPtr, TensorPtr)> loss_fn,
    int epochs,
    bool verbose = true
);
