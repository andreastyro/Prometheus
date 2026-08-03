#pragma once
#include "ml/tensor.hpp"
#include <string>
#include <vector>

// Save model parameters to a binary file.
void save(const std::string& path, std::vector<TensorPtr> params);

// Load model parameters from a file saved with save().
std::vector<TensorPtr> load(const std::string& path);

// ── Checkpointing ─────────────────────────────────────────────────────────────
// A checkpoint bundles model weights + the current epoch + the best loss into a
// single binary file so a training run can be resumed exactly where it left off.
//
// For optimizer state (Adam/AdamW momentum buffers), call optimizer.save_state()
// alongside save_checkpoint() and optimizer.load_state() alongside
// load_checkpoint() — they write a companion file you can name "run.opt.bin".
//
// Format: magic "CKPT" + version(int) + epoch(int) + loss(float)
//         + same block layout as save() for the parameter tensors.

struct Checkpoint {
    int   epoch;
    float loss;
};

void       save_checkpoint(const std::string& path,
                           const std::vector<TensorPtr>& params,
                           int epoch, float loss);

Checkpoint load_checkpoint(const std::string& path,
                           std::vector<TensorPtr>& params);
