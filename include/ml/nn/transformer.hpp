#pragma once
#include "ml/nn/module.hpp"
#include "ml/nn/attention.hpp"
#include "ml/nn/layernorm.hpp"
#include "ml/nn/linear.hpp"

// TransformerBlock — one complete transformer layer.
//
// Stacks all the components of a standard transformer encoder block:
//
//   1. Multi-Head Self-Attention
//      Each token attends to all other tokens to gather context.
//
//   2. Residual connection + LayerNorm  (Add & Norm)
//      Add the attention output back to the input (residual), then normalise.
//      The residual lets gradients flow directly from top to bottom,
//      making very deep networks trainable.
//
//   3. Feedforward network (FFN)
//      Two linear layers with ReLU in between: embed_dim -> ff_dim -> embed_dim.
//      Applied independently to each token position. Typically ff_dim = 4 * embed_dim.
//      This is where most of the model's "thinking" capacity lives.
//
//   4. Second residual + LayerNorm
//
// Full forward pass:
//   x = norm1(x + attention(x))
//   x = norm2(x + ff2(relu(ff1(x))))
//
// Stack multiple TransformerBlocks in a Sequential to build a full transformer.
//
// Input:  [seq_len, embed_dim]
// Output: [seq_len, embed_dim]
class TransformerBlock : public Module {
public:
    MultiHeadAttention attn;  // self-attention over the sequence
    LayerNorm          norm1; // normalise after attention + residual
    LayerNorm          norm2; // normalise after feedforward + residual
    Linear             ff1;   // first feedforward linear: embed_dim -> ff_dim
    Linear             ff2;   // second feedforward linear: ff_dim -> embed_dim

    int embed_dim;

    // ff_dim: inner dimension of the feedforward network, typically 4 * embed_dim
    TransformerBlock(int embed_dim, int num_heads, int ff_dim);

    TensorPtr forward(TensorPtr input) override;

    // Collects parameters from attn, norm1, norm2, ff1, ff2 (12 tensors total)
    std::vector<TensorPtr> parameters() override;
};
