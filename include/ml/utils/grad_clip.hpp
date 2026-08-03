#pragma once
#include <vector>
#include "ml/tensor.hpp"

/// Clips the global gradient norm across all parameters in-place.
///
/// Computes the L2 norm of all gradients concatenated, then scales every
/// gradient down by (max_norm / total_norm) if the norm exceeds max_norm.
/// Monitoring the returned norm tells you when clipping is active.
///
/// @param params   The list of parameter tensors (same list you pass to an optimizer)
/// @param max_norm Maximum allowed gradient norm
/// @returns        The gradient norm BEFORE clipping
float clip_grad_norm(std::vector<TensorPtr>& params, float max_norm);
