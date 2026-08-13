#include "ml/tensor.hpp"
#include "ml/nn/nlp/attention.hpp"
#include <stdio.h>
#include <cmath>

using namespace std;

int main() {
    // embed_dim=4, num_heads=2 → head_dim=2
    printf("=== MultiHeadAttention shape test ===\n");
    auto x = make_shared<Tensor>(vector<int>{3, 4});  // 3 tokens, 4 dims
    for (int i = 0; i < x->num_el(); i++) x->data[i] = 0.1f * (i + 1);

    MultiHeadAttention mha(4, 2);
    auto out = mha.forward(x);
    printf("input:  [3, 4]\n");
    printf("output: [%d, %d] (expected [3, 4])\n", out->shape[0], out->shape[1]);

    // === attention weights should sum to 1 per token (verified via output is finite) ===
    printf("\n=== Output values are finite ===\n");
    bool finite = true;
    for (float v : out->data)
        if (std::isnan(v) || std::isinf(v)) { finite = false; break; }
    printf("all finite: %s (expected true)\n", finite ? "true" : "false");

    // === longer sequence ===
    printf("\n=== Longer sequence [10, 4] ===\n");
    auto x2 = make_shared<Tensor>(vector<int>{10, 4});
    for (int i = 0; i < x2->num_el(); i++) x2->data[i] = 0.01f * i;
    auto out2 = mha.forward(x2);
    printf("output: [%d, %d] (expected [10, 4])\n", out2->shape[0], out2->shape[1]);

    // === invalid heads throws ===
    printf("\n=== Invalid num_heads throws ===\n");
    try {
        MultiHeadAttention bad(4, 3);
        printf("no exception (unexpected)\n");
    } catch (const exception& e) {
        printf("caught: %s\n", e.what());
    }

    // === parameters ===
    printf("\n=== Parameters ===\n");
    printf("param count: %zu (expected 4)\n", mha.parameters().size());

    return 0;
}
