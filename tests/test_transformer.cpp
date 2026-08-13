#include "ml/tensor.hpp"
#include "ml/nn/nlp/positional_encoding.hpp"
#include "ml/nn/nlp/transformer.hpp"
#include <stdio.h>
#include <cmath>

using namespace std;

int main() {
    // === PositionalEncoding ===
    printf("=== PositionalEncoding ===\n");
    auto x = make_shared<Tensor>(vector<int>{3, 4});  // 3 tokens, 4 dims
    for (int i = 0; i < x->num_el(); i++) x->data[i] = 0.0f;

    PositionalEncoding pe(100, 4);
    auto pe_out = pe.forward(x);
    printf("output shape: [%d, %d] (expected [3, 4])\n", pe_out->shape[0], pe_out->shape[1]);

    // each position should produce a different encoding
    bool all_different = true;
    for (int d = 0; d < 4; d++)
        if (pe_out->data[0 * 4 + d] == pe_out->data[1 * 4 + d])
            { all_different = false; break; }
    printf("positions produce different encodings: %s (expected true)\n",
        all_different ? "true" : "false");

    // position 0 dim 0 should be sin(0) = 0
    printf("pos 0 dim 0 (expected 0.0000): %.4f\n", pe_out->data[0]);
    // position 1 dim 0 should be sin(1) = 0.8415
    printf("pos 1 dim 0 (expected 0.8415): %.4f\n", pe_out->data[4]);

    // === TransformerBlock ===
    printf("\n=== TransformerBlock ===\n");
    auto x2 = make_shared<Tensor>(vector<int>{3, 4});
    for (int i = 0; i < x2->num_el(); i++) x2->data[i] = 0.1f * (i + 1);

    // embed_dim=4, num_heads=2, ff_dim=8
    TransformerBlock block(4, 2, 8);
    auto out = block.forward(x2);
    printf("input:  [3, 4]\n");
    printf("output: [%d, %d] (expected [3, 4])\n", out->shape[0], out->shape[1]);

    bool finite = true;
    for (float v : out->data)
        if (std::isnan(v) || std::isinf(v)) { finite = false; break; }
    printf("output is finite: %s (expected true)\n", finite ? "true" : "false");

    // === Parameter count ===
    // attn: 4, norm1: 2, norm2: 2, ff1: 2, ff2: 2 = 12 total
    printf("\nparam count: %zu (expected 12)\n", block.parameters().size());

    return 0;
}
