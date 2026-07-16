#pragma once
#include "ml/tensor.hpp"
#include <cmath>

// DataSplit — holds the output of a train/val/test split.
// Use with data_split() below.
struct DataSplit {
    TensorPtr x_train, y_train; // training inputs and labels
    TensorPtr x_val,   y_val;   // validation inputs and labels (used to monitor overfitting)
    TensorPtr x_test,  y_test;  // test inputs and labels (held out until final evaluation)
};

// DataLoader — serves data in mini-batches during training.
//
// Instead of passing the whole dataset to the model at once (expensive)
// or one sample at a time (noisy gradients), mini-batching processes a
// small fixed chunk each step. This balances gradient quality and speed.
//
// Usage:
//   DataLoader loader(x, y, batch_size=32, shuffle=true);
//   while (loader.has_next()) {
//       auto [xb, yb] = loader.next_batch();
//       // train on xb, yb
//   }
//   loader.reset(); // start the next epoch
class DataLoader {
public:
    TensorPtr x;    // all input data
    TensorPtr y;    // all target labels
    int batch_size; // number of samples per batch
    bool shuffle;   // whether to randomise sample order each epoch

    int index; // current position in the dataset

    DataLoader(TensorPtr x, TensorPtr y, int batch_size, bool shuffle = false);

    // Return the next batch of (x, y). Advances the internal index by batch_size.
    std::pair<TensorPtr, TensorPtr> next_batch();

    // True if there are still unprocessed samples this epoch
    bool has_next();

    // Reset to the start of the dataset (call at the beginning of each epoch)
    void reset();

    // Randomly reorder the samples (call after reset() if shuffle=true)
    void reshuffle();

    // Total number of batches per epoch (ceiling division of dataset size / batch_size)
    int size() { return (int)std::ceil((float)x->shape[0] / batch_size); }
};

// Split a dataset into train/val/test subsets.
// Ratios must sum to 1.0. val_ratio defaults to 0 (no validation set).
// Set shuffle=true to randomise before splitting.
DataSplit data_split(TensorPtr x, TensorPtr y,
                     float train_ratio, float test_ratio,
                     float val_ratio = 0.0f, bool shuffle = false);
