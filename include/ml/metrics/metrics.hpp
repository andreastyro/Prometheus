#pragma once
#include "ml/tensor.hpp"
#include "ml/nn/module.hpp"

// ── Classification metrics ────────────────────────────────────────────────
// These assume binary classification: pred and target contain 0s and 1s.
// pred should be rounded predictions (not raw probabilities).

// Fraction of samples predicted correctly: (TP + TN) / total
float accuracy(TensorPtr pred, TensorPtr target);

// Of all samples predicted positive, what fraction actually were?
// precision = TP / (TP + FP)
// High precision = few false alarms
float precision(TensorPtr pred, TensorPtr target);

// Of all actual positive samples, what fraction did we catch?
// recall = TP / (TP + FN)
// High recall = few missed positives
float recall(TensorPtr pred, TensorPtr target);

// Harmonic mean of precision and recall: 2 * (P * R) / (P + R)
// Single number that balances both. 1.0 = perfect, 0.0 = worst.
float f1_score(TensorPtr pred, TensorPtr target);

// 2x2 table of [TP, FP, FN, TN] showing where predictions went right and wrong
// Returned as a tensor of shape [2, 2]:
//   [[TP, FP],
//    [FN, TN]]
TensorPtr confusion_matrix(TensorPtr pred, TensorPtr target);

// ── Regression metrics ────────────────────────────────────────────────────

// R² (coefficient of determination) — how much of the variance in y the model explains.
// 1.0 = perfect, 0.0 = as good as predicting the mean, negative = worse than the mean.
float r2_score(TensorPtr pred, TensorPtr target);

// ── Prediction helper ─────────────────────────────────────────────────────

// Run the model and return class predictions (argmax of output).
// Useful for multi-class classification where output is softmax probabilities.
TensorPtr predict(Module& model, TensorPtr x);
