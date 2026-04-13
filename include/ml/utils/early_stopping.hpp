#pragma once

struct EarlyStopping {
    int patience;
    float min_delta;
    int counter;
    float best_loss;
    bool should_stop;

    EarlyStopping(int patience = 5, float min_delta = 0.0f);
    bool step(float val_loss);
    void reset();
};
