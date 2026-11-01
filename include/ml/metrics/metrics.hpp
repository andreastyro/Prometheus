#pragma once
#include "ml/tensor.hpp"
#include "ml/nn/module.hpp"

// ── Classification metrics ────────────────────────────────────────────────
// These assume binary classification: pred and target contain 0s and 1s.
// pred should be rounded predictions (not raw probabilities).

// Fraction of samples predicted correctly: (TP + TN) / total
float accuracy(TensorPtr pred, TensorPtr target);

// Of all samples predicted as a class, what fraction actually were?
//
// MACRO-AVERAGED across classes, matching sklearn's average="macro":
// precision is computed per class as TP_c / (TP_c + FP_c) and then averaged
// with equal weight. For a binary problem this averages the precision of the
// negative class and the positive class, so it is NOT the single-class
// TP / (TP + FP) figure.
float precision(TensorPtr pred, TensorPtr target);

// Of all actual samples of a class, what fraction did we catch?
// Also MACRO-AVERAGED: TP_c / (TP_c + FN_c) per class, then averaged.
float recall(TensorPtr pred, TensorPtr target);

// Harmonic mean of the macro-averaged precision and recall:
// 2 * (P_macro * R_macro) / (P_macro + R_macro).
//
// Note this is the F1 *of the averages*, not the average of the per-class F1
// scores, which is what sklearn's f1_score(average="macro") returns. The two
// differ when the classes are imbalanced.
float f1_score(TensorPtr pred, TensorPtr target);

// Counts of predictions against truth, shape [n_classes, n_classes].
// Rows are the TRUE class, columns are the PREDICTED class, so entry [t][p] is
// the number of samples of true class t that were predicted as p. This is the
// sklearn orientation. For a binary problem that means:
//   [[TN, FP],
//    [FN, TP]]
// so the true positives sit at [1][1], not [0][0].
TensorPtr confusion_matrix(TensorPtr pred, TensorPtr target);

// ── Regression metrics ────────────────────────────────────────────────────

// R² (coefficient of determination) — how much of the variance in y the model explains.
// 1.0 = perfect, 0.0 = as good as predicting the mean, negative = worse than the mean.
float r2_score(TensorPtr pred, TensorPtr target);

// ── Prediction helper ─────────────────────────────────────────────────────

// Run the model and return class predictions (argmax of output).
// Useful for multi-class classification where output is softmax probabilities.
TensorPtr predict(Module& model, TensorPtr x);
