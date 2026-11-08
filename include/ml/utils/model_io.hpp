#pragma once
#include "ml/tensor.hpp"
#include <map>
#include <string>
#include <utility>
#include <vector>

// ── File format ──────────────────────────────────────────────────────────────
//
// Models and checkpoints are written as safetensors -- the layout Hugging Face
// and most of the ecosystem use: an 8-byte little-endian header length, a JSON
// header giving each tensor's name, dtype, shape and byte range, then the raw
// row-major bytes. A file written here opens in PyTorch, JAX or numpy without
// this library, and weights published as safetensors (GPT-2 and nearly
// everything on the Hugging Face hub) load here. The reader and writer are
// this library's own; nothing is linked.
//
// Every dtype round-trips: F32, F64, F16, BF16, I64, I32, BOOL. Files in the
// previous formats -- the plain model file and the "CKPT" checkpoint, both
// float32 only -- still load.

using NamedTensors = std::vector<std::pair<std::string, TensorPtr>>;

/// Write named tensors as safetensors. A view, or any dtype, is written as its
/// values in row-major order. Names must be unique and non-empty.
void save_safetensors(const std::string& path, const NamedTensors& tensors,
                      const std::map<std::string, std::string>& metadata = {});

/// Read a safetensors file, tensors in the order their bytes appear. The
/// header's free-form "__metadata__" goes into `metadata` when given.
NamedTensors load_safetensors(const std::string& path,
                              std::map<std::string, std::string>* metadata = nullptr);

/// Save model parameters in order, named "000000", "000001", ...
void save(const std::string& path, std::vector<TensorPtr> params);

/// Load parameters written by save(), in either format.
std::vector<TensorPtr> load(const std::string& path);

// ── Checkpointing ─────────────────────────────────────────────────────────────
// The parameters plus the epoch and best loss, so a run can resume where it
// left off: a safetensors file with the two numbers in its metadata. For Adam
// and AdamW state, call optimizer.save_state() alongside.
struct Checkpoint {
    int   epoch;
    float loss;
};

void       save_checkpoint(const std::string& path,
                           const std::vector<TensorPtr>& params,
                           int epoch, float loss);
Checkpoint load_checkpoint(const std::string& path,
                           std::vector<TensorPtr>& params);
