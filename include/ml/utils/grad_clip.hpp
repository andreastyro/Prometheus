#pragma once
#include <vector>
#include "ml/tensor.hpp"

/// Clips the global gradient norm across all parameters in-place.
///
/// Computes the L2 norm of all gradients concatenated, then scales every
/// gradient down by (max_norm / total_norm) if the norm exceeds max_norm.
/// Monitoring the returned norm tells you when clipping is active.
///
/// @param params      The list of parameter tensors (same list you pass to an optimizer)
/// @param max_norm    Maximum allowed gradient norm
/// @param return_norm On the GPU, reading the norm back makes the CPU wait for
///                    the whole backward pass before it can queue the optimizer
///                    step (~1.5 ms of a GPT step on a GTX 970). false clips on
///                    the device without that wait and returns NaN.
/// @returns           The gradient norm BEFORE clipping (NaN if not read)
float clip_grad_norm(std::vector<TensorPtr>& params, float max_norm, bool return_norm = true);
