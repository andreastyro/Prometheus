#pragma once
#include "ml/tensor.hpp"
#include <vector>

/// Mean Squared Error: average of (pred - target)^2
/// Penalises large errors heavily. The most common regression loss.
TensorPtr mse_loss(TensorPtr pred, TensorPtr target);

/// Mean Absolute Error: average of |pred - target|
/// Less sensitive to outliers than MSE — does not square the error.
TensorPtr mae_loss(TensorPtr pred, TensorPtr target);

/// Huber loss: MSE for small errors, MAE for large errors.
/// Errors below delta use MSE (smooth), above delta use MAE (robust to outliers).
/// @param delta  threshold between MSE and MAE regions (default 1.0)
TensorPtr huber_loss(TensorPtr pred, TensorPtr target, float delta = 1.0f);

/// Binary Cross-Entropy: -(y * log(p) + (1-y) * log(1-p))
/// Use with Sigmoid output for binary classification (predicting 0 or 1).
TensorPtr bce_loss(TensorPtr pred, TensorPtr target);

/// Cross-Entropy: penalises the model for assigning low probability to the correct class.
/// Use with Softmax output for multi-class classification.
TensorPtr cross_entropy_loss(TensorPtr pred, TensorPtr target);

/// KL Divergence: measures how different distribution q is from reference p.
/// Result is zero when p == q, positive otherwise.
/// Used in VAEs and knowledge distillation.
TensorPtr kl_divergence(TensorPtr p, TensorPtr q);

/// Reconstruction loss: sum of squared differences between input and output.
/// Used in autoencoders — penalises the model for not reconstructing the input.
TensorPtr reconstruction_loss(TensorPtr input, TensorPtr output);

/// Contrastive (triplet) loss: pulls anchor and positive together,
/// pushes anchor and negative apart by at least margin.
/// Used to train embeddings where similar items should cluster together.
/// @param margin  minimum distance between anchor-negative pairs (default 1.0)
TensorPtr contrastive_loss(TensorPtr anchor, TensorPtr positive, TensorPtr negative, float margin = 1.0f);

/// L1 regularisation: sum(|w|) * lambda — encourages sparse weights (many become zero).
/// Add to the main loss to penalise large weights and reduce overfitting.
TensorPtr l1_regularization(std::vector<TensorPtr> params, float lambda_);

/// L2 regularisation: sum(w^2) * lambda — penalises large weights, keeps them small.
/// Add to the main loss to penalise large weights and reduce overfitting.
TensorPtr l2_regularization(std::vector<TensorPtr> params, float lambda_);

/// Sparse cross-entropy for a single position.
/// logits: raw unnormalised scores [vocab_size] (NOT softmaxed)
/// target_idx: the correct class index (integer)
/// Numerically stable via log-sum-exp. Backward: softmax(logits) - one_hot(target)
TensorPtr cross_entropy_sparse(TensorPtr logits, int target_idx);

/// Sparse cross-entropy over a full sequence in one call.
/// logits: [seq_len, vocab_size] raw scores
/// targets: vector of seq_len correct class indices
/// Returns mean loss over all positions.
TensorPtr cross_entropy_sparse_seq(TensorPtr logits, const std::vector<int>& targets);
