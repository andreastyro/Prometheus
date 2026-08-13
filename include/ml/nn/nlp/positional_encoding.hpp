#pragma once
#include "ml/tensor.hpp"

// PositionalEncoding — injects position information into token embeddings.
//
// Attention has no built-in sense of order — "the cat sat" and "sat cat the"
// look the same to attention if the tokens are identical. Positional encoding
// fixes this by adding a unique signal to each position.
//
// The encoding uses sine and cosine waves at different frequencies:
//   PE[pos, 2i]   = sin(pos / 10000^(2i / embed_dim))
//   PE[pos, 2i+1] = cos(pos / 10000^(2i / embed_dim))
//
// Think of it like an analog clock with many hands running at different speeds:
//   - Early dimensions (small i) change rapidly with position  (fast hand)
//   - Late dimensions  (large i) change slowly                 (slow hand)
// Together, every position gets a unique combination of values — like binary
// digits, but smooth and differentiable.
//
// This is NOT a Module — it has no learnable parameters and is not in the
// computation graph. Call forward() to add the encoding to your embeddings.
//
// Input:  [seq_len, embed_dim]
// Output: [seq_len, embed_dim]  (input + positional signal)
class PositionalEncoding {
public:
    int max_len;   // maximum sequence length this encoding is pre-computed for
    int embed_dim; // must match the embedding dimension of your model

    PositionalEncoding(int max_len, int embed_dim);

    // Add positional encoding to input. Does not affect the gradient graph.
    TensorPtr forward(TensorPtr input);
};
