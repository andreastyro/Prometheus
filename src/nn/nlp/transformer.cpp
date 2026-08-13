#include "ml/nn/nlp/transformer.hpp"
#include "ml/ops.hpp"

using namespace std;

TransformerBlock::TransformerBlock(int embed_dim, int num_heads, int ff_dim, bool causal, bool rope)
    : embed_dim(embed_dim),
      attn(embed_dim, num_heads, causal, rope),
      norm1(embed_dim),
      norm2(embed_dim),
      ff1(embed_dim, ff_dim),
      ff2(ff_dim, embed_dim) {}

// Pre-norm architecture (GPT-2 style): normalise before each sub-layer, not after.
// This is more stable at depth and is the standard for modern language models.
// FFN uses GELU instead of ReLU — the standard in GPT-2/3/4.
TensorPtr TransformerBlock::forward(TensorPtr input) {
    // Attention sub-layer: x = input + attn(norm1(input))
    auto x = add(input, attn.forward(norm1.forward(input)));
    // FFN sub-layer:       x = x + ff2(gelu(ff1(norm2(x))))
    return add(x, ff2.forward(gelu(ff1.forward(norm2.forward(x)))));
}

vector<TensorPtr> TransformerBlock::parameters() {
    vector<TensorPtr> params;
    for (auto& p : attn.parameters())  params.push_back(p);
    for (auto& p : norm1.parameters()) params.push_back(p);
    for (auto& p : norm2.parameters()) params.push_back(p);
    for (auto& p : ff1.parameters())   params.push_back(p);
    for (auto& p : ff2.parameters())   params.push_back(p);
    return params;
}
