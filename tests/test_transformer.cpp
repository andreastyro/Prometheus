#include "ml/tensor.hpp"
#include "ml/nn/nlp/positional_encoding.hpp"
#include "ml/nn/nlp/embedding.hpp"
#include "ml/utils/testing.hpp"
#include "ml/nn/nlp/transformer.hpp"
#include <stdio.h>
#include <cmath>

using namespace std;

int main() {
    // === PositionalEncoding ===
    printf("=== PositionalEncoding ===\n");
    auto x = make_shared<Tensor>(Shape{3, 4});  // 3 tokens, 4 dims
    for (int i = 0; i < x->num_el(); i++) x->data()[i] = 0.0f;

    PositionalEncoding pe(100, 4);
    auto pe_out = pe.forward(x);
    printf("output shape: [%d, %d] (expected [3, 4])\n", pe_out->shape()[0], pe_out->shape()[1]);

    // each position should produce a different encoding
    bool all_different = true;
    for (int d = 0; d < 4; d++)
        if (pe_out->data()[0 * 4 + d] == pe_out->data()[1 * 4 + d])
            { all_different = false; break; }
    CHECK_TRUE("positions produce different encodings", all_different);

    // position 0 dim 0 should be sin(0) = 0
    CHECK_NEAR("pos 0 dim 0", pe_out->data()[0], 0.0f, 1e-4f);
    // position 1 dim 0 should be sin(1) = 0.8415
    CHECK_NEAR("pos 1 dim 0", pe_out->data()[4], 0.8415f, 1e-4f);

    // The encoding is a constant, so the gradient must pass straight through.
    // Regression test: this previously returned a tensor with no grad_fn, which
    // silently severed the graph and left upstream embeddings untrained.
    {
        Embedding emb(10, 4);
        auto ids = make_shared<Tensor>(Shape{1, 3});
        ids->data()[0] = 1; ids->data()[1] = 5; ids->data()[2] = 7;

        PositionalEncoding pe_g(16, 4);
        auto y = pe_g.forward(emb.forward(ids));
        y->backward();

        float total = 0.0f;
        for (float g : emb.weight->grad_values()) total += g;
        // 3 tokens x 4 dims, each seeded with dL/dy = 1
        CHECK_NEAR("gradient reaches embedding", total, 12.0f, 1e-4f);
    }

    // === TransformerBlock ===
    printf("\n=== TransformerBlock ===\n");
    auto x2 = make_shared<Tensor>(Shape{3, 4});
    for (int i = 0; i < x2->num_el(); i++) x2->data()[i] = 0.1f * (i + 1);

    // embed_dim=4, num_heads=2, ff_dim=8
    TransformerBlock block(4, 2, 8);
    auto out = block.forward(x2);
    printf("input:  [3, 4]\n");
    printf("output: [%d, %d] (expected [3, 4])\n", out->shape()[0], out->shape()[1]);

    bool finite = true;
    for (float v : out->values())
        if (std::isnan(v) || std::isinf(v)) { finite = false; break; }
    CHECK_TRUE("output is finite", finite);

    // === Parameter count ===
    // attn: 4, norm1: 2, norm2: 2, ff1: 2, ff2: 2 = 12 total
    CHECK_NEAR("param count", (double)block.parameters().size(), 12.0, 0.0);

    return test_report();
}
