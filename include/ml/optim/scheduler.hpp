#pragma once
#include <cmath>

// LR Schedulers — automatically adjust the learning rate during training.
//
// Why? A large learning rate early on helps explore the loss landscape quickly.
// A smaller rate later helps fine-tune into a precise minimum.
// Schedulers automate this decay so you don't have to change it by hand.
//
// Usage:
//   scheduler.step() returns the new lr each epoch.
//   Assign it to optimizer.lr to apply it.
//   e.g.:
//     optimizer.lr = scheduler.step();

// Base class — all schedulers inherit from this
class LRScheduler {
public:
    float base_lr;      // the starting learning rate
    int   step_count;   // how many times step() has been called so far

    LRScheduler(float base_lr) : base_lr(base_lr), step_count(0) {}
    virtual ~LRScheduler() {}

    // Advance one epoch and return the new learning rate
    virtual float step() = 0;

    // Peek at the current learning rate without advancing
    float get_lr() const { return base_lr; }
};

// StepLR — multiply lr by gamma every step_size epochs.
//
// Example: StepLR(0.01, step_size=50, gamma=0.5)
//   epoch 1-50:  lr = 0.01
//   epoch 51-100: lr = 0.005
//   epoch 101-150: lr = 0.0025
//
// Good starting point — simple and predictable.
class StepLR : public LRScheduler {
public:
    int   step_size;  // number of epochs between each decay
    float gamma;      // multiplicative decay factor (< 1 to reduce lr)
    float current_lr; // lr after the most recent step

    StepLR(float base_lr, int step_size, float gamma = 0.1f)
        : LRScheduler(base_lr), step_size(step_size), gamma(gamma), current_lr(base_lr) {}

    float step() override {
        step_count++;
        if (step_count % step_size == 0)
            current_lr *= gamma; // decay at each milestone
        return current_lr;
    }
};

// ExponentialLR — multiply lr by gamma every single epoch.
//
// Smoother than StepLR — lr decays continuously rather than in steps.
// With gamma=0.95, lr halves roughly every 14 epochs.
class ExponentialLR : public LRScheduler {
public:
    float gamma;      // per-epoch decay factor (typically 0.9 – 0.99)
    float current_lr;

    ExponentialLR(float base_lr, float gamma = 0.95f)
        : LRScheduler(base_lr), gamma(gamma), current_lr(base_lr) {}

    float step() override {
        step_count++;
        current_lr *= gamma;
        return current_lr;
    }
};

// CosineAnnealingLR — smoothly oscillates lr between base_lr and min_lr
// following a cosine curve over T_max epochs.
//
// Starts at base_lr, gently drops to min_lr at epoch T_max, then restarts.
// The smooth shape avoids abrupt drops and often finds better minima than step decay.
// Popular in computer vision and NLP.
//
// Formula: lr = min_lr + 0.5*(base_lr - min_lr) * (1 + cos(pi * t / T_max))
class CosineAnnealingLR : public LRScheduler {
public:
    float min_lr; // floor — lr will not go below this
    int   T_max;  // half-cycle length in epochs

    CosineAnnealingLR(float base_lr, int T_max, float min_lr = 0.0f)
        : LRScheduler(base_lr), min_lr(min_lr), T_max(T_max) {}

    float step() override {
        // t cycles within [0, T_max) — restart every T_max epochs
        float t = static_cast<float>(step_count % T_max);
        step_count++;
        return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + std::cos(3.14159265f * t / T_max));
    }
};
