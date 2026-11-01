#include "ml/tensor.hpp"
#include "ml/utils/testing.hpp"
#include "ml/nn/nlp/embedding.hpp"
#include <stdio.h>

using namespace std;

int main() {
    // vocab_size=5, embed_dim=4
    Embedding emb(5, 4);

    // === Shape test ===
    // input: [batch=2, seq_len=3] — token ids as floats
    printf("=== Embedding shape test ===\n");
    auto ids = make_shared<Tensor>(vector<int>{2, 3});
    // batch 0: tokens [0, 1, 2], batch 1: tokens [3, 4, 0]
    ids->data = {0, 1, 2, 3, 4, 0};

    auto out = emb.forward(ids);
    printf("input:  [2, 3]\n");
    CHECK_NEAR("output dim 0", (double)(out->shape[0]), 2, 0.0);
CHECK_NEAR("output dim 1", (double)(out->shape[1]), 3, 0.0);
CHECK_NEAR("output dim 2", (double)(out->shape[2]), 4, 0.0);

    // === Same token id produces same vector ===
    printf("\n=== Same id same vector ===\n");
    // ids[0][0] = 0, ids[1][2] = 0 — both should produce the same embedding
    bool same = true;
    for (int d = 0; d < 4; d++) {
        float v1 = out->data[0 * 4 + d];        // batch 0, token 0
        float v2 = out->data[(1 * 3 + 2) * 4 + d]; // batch 1, token 2 (id=0)
        if (v1 != v2) { same = false; break; }
    }
    printf("token id 0 gives same vector in both positions: %s (expected true)\n", same ? "true" : "false");

    // === Parameters ===
    printf("\n=== Parameters ===\n");
    auto params = emb.parameters();
    printf("param count: %zu (expected 1)\n", params.size());
    auto& w = params[0];
    CHECK_NEAR("weight shape dim 0", (double)(w->shape[0]), 5, 0.0);
CHECK_NEAR("weight shape dim 1", (double)(w->shape[1]), 4, 0.0);

    // === Out of range throws ===
    printf("\n=== Out of range id ===\n");
    auto bad_ids = make_shared<Tensor>(vector<int>{1, 1});
    bad_ids->data = {99};
    try {
        emb.forward(bad_ids);
        printf("no exception thrown (unexpected)\n");
    } catch (const exception& e) {
        printf("caught exception: %s\n", e.what());
    }

    return test_report();
}
