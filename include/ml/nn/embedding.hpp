#pragma once
#include "ml/nn/module.hpp"

/// Maps integer token IDs to dense float vectors (a learned lookup table).
///
/// Each word in your vocabulary gets its own row in the weight matrix.
/// When the model sees token ID 42, it returns row 42 of the weight matrix.
///
/// Why vectors instead of raw integers?
///   Integers have no useful geometry. Embedding vectors live in a continuous
///   space where similar words end up near each other after training —
///   "king" and "queen" have similar vectors, "king" and "car" do not.
///
/// The weight matrix starts random and trains like any other parameter.
/// Gradients update only the rows that were looked up in that forward pass.
///
/// Input:  [batch, seq_len]             — token IDs stored as floats
/// Output: [batch, seq_len, embed_dim]  — one vector per token
class Embedding : public Module {
public:
    TensorPtr weight; ///< [vocab_size, embed_dim] — the full lookup table

    int vocab_size; ///< Number of unique tokens (size of the vocabulary)
    int embed_dim;  ///< Length of each token's vector representation

    /// @param vocab_size  number of unique tokens (e.g. 10000)
    /// @param embed_dim   vector size per token (e.g. 128 or 512)
    Embedding(int vocab_size, int embed_dim);

    /// Look up the embedding vector for each token ID in the input.
    /// Token IDs are stored as floats and cast to int at lookup time.
    /// Throws std::runtime_error if any ID is outside [0, vocab_size).
    TensorPtr forward(TensorPtr input) override;

    std::vector<TensorPtr> parameters() override; ///< Returns {weight}
};
