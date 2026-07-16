#pragma once

// EarlyStopping — stops training when the model stops improving.
//
// Monitors the validation loss after each epoch. If it hasn't improved
// by at least min_delta for `patience` consecutive epochs, sets
// should_stop=true so you can break out of your training loop.
//
// This prevents overfitting: the model starts memorising the training set
// instead of learning general patterns, which shows up as the val loss
// flattening or rising while training loss keeps dropping.
//
// Usage:
//   EarlyStopping es(patience=10, min_delta=0.001f);
//   for (int epoch = 0; epoch < max_epochs; epoch++) {
//       float val_loss = evaluate(model, val_data);
//       if (es.step(val_loss)) break;  // stop if no improvement
//   }
struct EarlyStopping {
    int   patience;     // how many epochs to wait with no improvement before stopping
    float min_delta;    // minimum change that counts as an improvement
    int   counter;      // how many epochs since the last improvement
    float best_loss;    // lowest validation loss seen so far
    bool  should_stop;  // set to true when patience is exhausted

    EarlyStopping(int patience = 5, float min_delta = 0.0f);

    // Feed the current epoch's validation loss. Returns true when training should stop.
    bool step(float val_loss);

    // Reset to initial state (e.g. to reuse across multiple training runs)
    void reset();
};
