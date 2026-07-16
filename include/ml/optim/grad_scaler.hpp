#pragma once
#include "ml/tensor.hpp"
#include <vector>
#include <cmath>

// GradScaler — enables mixed precision training.
//
// Mixed precision trains with float16 (half) weights/activations for speed
// and memory savings, but float16 has a limited range: gradients that are
// very small can become zero ("underflow") and gradients that are large
// can become infinity ("overflow").
//
// The scaler solves this by:
//   1. Multiplying the loss by a large scale factor before backward()
//      This pushes small gradients out of the underflow range.
//   2. After backward(), dividing all gradients back down by the same factor
//      before the optimizer step (unscale).
//   3. Skipping the optimizer step if any gradient is inf or nan (overflow).
//   4. Dynamically adjusting the scale: grow it when training is clean,
//      shrink it immediately when overflow is detected.
//
// Typical usage:
//   scaled_loss = scaler.scale_loss(loss)       // multiply loss by scale
//   scaled_loss.backward()                       // backprop with inflated gradients
//   clean = scaler.unscale(model.parameters())  // divide grads back down
//   if (clean) optimizer.step()                  // only update if no overflow
//   scaler.update(!clean)                        // adjust scale for next iter
//   optimizer.zero_grad()
class GradScaler {
public:
    float scale;          // current loss multiplier (starts large, ~65536)
    float growth_factor;  // factor to multiply scale by when training is clean (e.g. 2)
    float backoff_factor; // factor to multiply scale by after overflow (e.g. 0.5)
    int   growth_interval;// how many clean steps before growing the scale
    int   step_count;     // number of steps since last scale growth
    bool  enabled;        // false = scaler is a no-op (useful for debugging)

    GradScaler(float init_scale      = 65536.0f,
               float growth_factor   = 2.0f,
               float backoff_factor  = 0.5f,
               int   growth_interval = 2000,
               bool  enabled         = true);

    // Return loss * scale. Call this before loss.backward().
    TensorPtr scale_loss(TensorPtr loss);

    // Divide every parameter's gradient by the current scale.
    // Returns false if any gradient contains inf or nan (overflow detected),
    // in which case the optimizer step should be skipped.
    bool unscale(std::vector<TensorPtr> params);

    // Update the scale based on whether overflow occurred this step.
    // had_overflow=true -> shrink scale by backoff_factor immediately
    // had_overflow=false -> grow scale by growth_factor every growth_interval steps
    void update(bool had_overflow);
};
