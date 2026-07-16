#pragma once
#include "ml/tensor.hpp"
#include <string>

// Load a CSV file and return it split into features (x) and labels (y).
//
// Args:
//   path:   path to the CSV file
//   y_col:  which column is the label. -1 means the last column (default).
//           Set to 0 for the first column.
//   header: if true, the first row is treated as column names and skipped.
//
// Returns a pair {x, y}:
//   x: [num_rows, num_feature_cols]  — all columns except y_col
//   y: [num_rows, 1]                 — the label column
//
// All values are read as floats. Categorical labels should be pre-encoded
// as integers before saving the CSV.
std::pair<TensorPtr, TensorPtr> read_csv(const std::string& path,
                                          int y_col = -1,
                                          bool header = true);
