#include "ml/nn/transformer.hpp"
#include "ml/ops.hpp"

using namespace std;

TransformerBlock::TransformerBlock(int embed_dim, int num_heads, int ff_dim)
    : embed_dim(embed_dim),
      attn(embed_dim, num_heads),
      norm1(embed_dim),
      norm2(embed_dim),
      ff1(embed_dim, ff_dim),
      ff2(ff_dim, embed_dim) {}

TensorPtr TransformerBlock::forward(TensorPtr input) {
    // --- attention sub-layer ---
    // attention output + residual
    auto attn_out = add(attn.forward(input), input);
    // layer norm
    auto x = norm1.forward(attn_out);

    // --- feedforward sub-layer ---
    // Linear → ReLU → Linear
    auto ff_out = ff2.forward(relu(ff1.forward(x)));
    // residual + layer norm
    return norm2.forward(add(ff_out, x));
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
