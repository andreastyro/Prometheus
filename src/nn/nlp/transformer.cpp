#include "../../ops_internal.hpp"
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
    if (input->device() == Device::CUDA) {
        // the same block with the work between gemms fused: residual + norm2,
        // bias + GELU, bias + residual -- one kernel each
        auto [x, h2] = cuda_ops::add_layernorm(input, attn.forward(norm1.forward(input)),
                                               norm2.gamma, norm2.beta, norm2.eps);
        auto h = cuda_ops::bias_gelu(matmul(h2, ff1.weights), ff1.bias);
        return cuda_ops::add_bias_residual(matmul(h, ff2.weights), ff2.bias, x);
    }
    // Attention sub-layer: x = input + attn(norm1(input)). Attention reads
    // the batch from the shape; everything else here is per-token over the
    // last dimension, so any leading dims pass straight through.
    auto x = add(input, attn.forward(norm1.forward(input)));
    // FFN sub-layer:       x = x + ff2(gelu(ff1(norm2(x))))
    // The residual is folded into ff2's gemm: one op instead of a gemm, a
    // bias pass and an add pass, each over the full [rows, embed] buffer.
    auto h = gelu(ff1.forward(norm2.forward(x)));
    return linear_op(h, ff2.weights, ff2.bias, x);
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
