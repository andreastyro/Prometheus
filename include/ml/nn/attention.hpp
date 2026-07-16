#pragma once
#include "ml/nn/module.hpp"

// MultiHeadAttention — the core mechanism behind transformer models.
//
// For each token in the sequence, attention answers: "which other tokens
// should I pay attention to right now, and how much?"
//
// The mechanism uses three learned projections of the input:
//   Q (Query)  — what am I looking for?
//   K (Key)    — what do I have to offer?
//   V (Value)  — what I will actually contribute if selected
//
// For each query position:
//   1. Compute dot products Q[i] · K[j] for all positions j  (similarity scores)
//   2. Scale by 1/sqrt(head_dim) to prevent dot products from growing too large
//   3. Apply softmax to get attention weights (sum to 1)
//   4. Take the weighted sum of V — the result is the attended representation
//
// Multi-head runs this process `num_heads` times in parallel, each head looking
// at a different aspect of the relationships. Outputs are concatenated, then
// projected back to embed_dim via W_o.
//
// All four weight matrices are square [embed_dim, embed_dim].
// head_dim = embed_dim / num_heads — each head sees a slice of the embedding.
//
// Input:  [seq_len, embed_dim]
// Output: [seq_len, embed_dim]
class MultiHeadAttention : public Module {
public:
    TensorPtr W_q; // [embed_dim, embed_dim] — query projection
    TensorPtr W_k; // [embed_dim, embed_dim] — key projection
    TensorPtr W_v; // [embed_dim, embed_dim] — value projection
    TensorPtr W_o; // [embed_dim, embed_dim] — output projection (combines all heads)

    int embed_dim; // total embedding dimension
    int num_heads; // number of attention heads to run in parallel
    int head_dim;  // dimension per head = embed_dim / num_heads

    // embed_dim must be divisible by num_heads
    MultiHeadAttention(int embed_dim, int num_heads);

    // Compute multi-head self-attention over the input sequence
    TensorPtr forward(TensorPtr input) override;

    // Returns {W_q, W_k, W_v, W_o}
    std::vector<TensorPtr> parameters() override;
};
