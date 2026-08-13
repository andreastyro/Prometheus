#include "ml/tensor.hpp"
#include "ml/nn/nlp/attention.hpp"
#include <stdio.h>
#include <cmath>

using namespace std;

static bool all_finite(const vector<float>& v) {
    for (float x : v) if (std::isnan(x) || std::isinf(x)) return false;
    return true;
}

static bool has_nonzero_grad(TensorPtr t) {
    for (float g : t->grad) if (g != 0.0f) return true;
    return false;
}

int main() {
    // embed_dim=4, num_heads=2 → head_dim=2
    printf("=== MultiHeadAttention shape test ===\n");
    auto x = make_shared<Tensor>(vector<int>{3, 4});
    for (int i = 0; i < x->num_el(); i++) x->data[i] = 0.1f * (i + 1);

    MultiHeadAttention mha(4, 2);
    auto out = mha.forward(x);
    printf("input:  [3, 4]\n");
    printf("output: [%d, %d] (expected [3, 4])\n", out->shape[0], out->shape[1]);

    printf("\n=== Output values are finite ===\n");
    printf("all finite: %s (expected true)\n", all_finite(out->data) ? "true" : "false");

    printf("\n=== Longer sequence [10, 4] ===\n");
    auto x2 = make_shared<Tensor>(vector<int>{10, 4});
    for (int i = 0; i < x2->num_el(); i++) x2->data[i] = 0.01f * i;
    auto out2 = mha.forward(x2);
    printf("output: [%d, %d] (expected [10, 4])\n", out2->shape[0], out2->shape[1]);

    printf("\n=== Invalid num_heads throws ===\n");
    try {
        MultiHeadAttention bad(4, 3);
        printf("no exception (unexpected)\n");
    } catch (const exception& e) {
        printf("caught: %s\n", e.what());
    }

    printf("\n=== Parameters ===\n");
    printf("param count: %zu (expected 4)\n", mha.parameters().size());

    // ── Backward: verify all weight matrices receive gradients ────────────────
    printf("\n=== Backward — gradient flow ===\n");
    MultiHeadAttention mha2(4, 2);
    auto x3 = make_shared<Tensor>(vector<int>{3, 4});
    x3->requires_grad = true;
    for (int i = 0; i < x3->num_el(); i++) x3->data[i] = 0.1f * (i + 1);

    auto out3 = mha2.forward(x3);
    // seed gradient = all ones
    for (float& g : out3->grad) g = 1.0f;
    out3->backward();

    printf("W_q has grad: %s (expected true)\n", has_nonzero_grad(mha2.W_q) ? "true" : "false");
    printf("W_k has grad: %s (expected true)\n", has_nonzero_grad(mha2.W_k) ? "true" : "false");
    printf("W_v has grad: %s (expected true)\n", has_nonzero_grad(mha2.W_v) ? "true" : "false");
    printf("W_o has grad: %s (expected true)\n", has_nonzero_grad(mha2.W_o) ? "true" : "false");
    printf("x  has grad: %s (expected true)\n",  has_nonzero_grad(x3)       ? "true" : "false");
    printf("grads finite: %s (expected true)\n",
        (all_finite(mha2.W_q->grad) && all_finite(mha2.W_k->grad) &&
         all_finite(mha2.W_v->grad) && all_finite(mha2.W_o->grad) &&
         all_finite(x3->grad)) ? "true" : "false");

    // ── Causal mask backward ──────────────────────────────────────────────────
    printf("\n=== Causal attention backward ===\n");
    MultiHeadAttention causal(4, 2, /*causal=*/true);
    auto xc = make_shared<Tensor>(vector<int>{4, 4});
    xc->requires_grad = true;
    for (int i = 0; i < xc->num_el(); i++) xc->data[i] = 0.05f * (i + 1);

    auto outc = causal.forward(xc);
    for (float& g : outc->grad) g = 1.0f;
    outc->backward();

    printf("W_q has grad: %s (expected true)\n", has_nonzero_grad(causal.W_q) ? "true" : "false");
    printf("grads finite: %s (expected true)\n",
        (all_finite(causal.W_q->grad) && all_finite(causal.W_k->grad) &&
         all_finite(causal.W_v->grad) && all_finite(causal.W_o->grad) &&
         all_finite(xc->grad)) ? "true" : "false");

    return 0;
}
