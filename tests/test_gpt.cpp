#include "ml/nn/nlp/gpt.hpp"
#include "ml/loss.hpp"
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
    // Small GPT: vocab=16, seq=4, embed=8, heads=2, layers=2
    int VOCAB = 16, MAXLEN = 32, EMBED = 8, HEADS = 2, LAYERS = 2;

    printf("=== GPT forward shape ===\n");
    GPT gpt(VOCAB, MAXLEN, EMBED, HEADS, LAYERS);
    auto token_ids = make_shared<Tensor>(vector<int>{4}, vector<float>{1.f, 3.f, 5.f, 2.f});
    auto logits = gpt.forward(token_ids);
    printf("logits shape: [%d, %d] (expected [4, %d])\n", logits->shape[0], logits->shape[1], VOCAB);
    printf("logits finite: %s (expected true)\n", all_finite(logits->data) ? "true" : "false");

    printf("\n=== cross_entropy_sparse_seq ===\n");
    vector<int> targets = {3, 5, 2, 1};
    auto loss = cross_entropy_sparse_seq(logits, targets);
    printf("loss shape: [%d] (expected [1])\n", loss->shape[0]);
    printf("loss value: %.4f (expected positive)\n", loss->data[0]);
    printf("loss finite: %s (expected true)\n", all_finite(loss->data) ? "true" : "false");

    printf("\n=== Backward — gradient flow ===\n");
    for (float& g : loss->grad) g = 1.0f;
    loss->backward();

    auto params_unused = gpt.parameters(); (void)params_unused;
    bool tok_emb_has_grad   = has_nonzero_grad(gpt.tok_emb.weight);
    bool pos_emb_has_grad   = has_nonzero_grad(gpt.pos_emb.weight);
    bool ln_f_has_grad      = has_nonzero_grad(gpt.ln_f.parameters()[0]);
    bool block0_wq_has_grad = has_nonzero_grad(gpt.blocks[0].attn.W_q);
    bool block0_ff1_has_grad= has_nonzero_grad(gpt.blocks[0].ff1.weights);

    printf("tok_emb.weight grad:    %s (expected true)\n", tok_emb_has_grad    ? "true" : "false");
    printf("pos_emb.weight grad:    %s (expected true)\n", pos_emb_has_grad    ? "true" : "false");
    printf("ln_f.scale grad:        %s (expected true)\n", ln_f_has_grad       ? "true" : "false");
    printf("block[0].W_q grad:      %s (expected true)\n", block0_wq_has_grad  ? "true" : "false");
    printf("block[0].ff1 grad:      %s (expected true)\n", block0_ff1_has_grad ? "true" : "false");

    printf("\n=== Weight tying ===\n");
    // tok_emb.weight grad should have contributions from BOTH the input embedding
    // AND the output projection (weight tying). Verify it's nonzero.
    printf("weight-tied grad nonzero: %s (expected true)\n",
           has_nonzero_grad(gpt.tok_emb.weight) ? "true" : "false");

    printf("\n=== Parameter count ===\n");
    auto p = gpt.parameters();
    int total = 0;
    for (auto& t : p) total += t->num_el();
    printf("total params: %d\n", total);
    // Expected: tok_emb [16,8]=128, pos_emb [32,8]=256, 2 blocks, ln_f
    // Each block: W_q,W_k,W_v,W_o [8,8]=64 each → 256, norm1+norm2 [8+8=16]*2=64, ff1[8,32]+ff2[32,8]=512
    // Total per block ≈ 832; 2 blocks ≈ 1664; + 128+256+16 = 2064
    printf("(no assertion — just informational)\n");

    printf("\n=== Invalid sequence length throws ===\n");
    try {
        auto too_long = make_shared<Tensor>(vector<int>{100}, vector<float>(100, 0.0f));
        gpt.forward(too_long);
        printf("no exception (unexpected)\n");
    } catch (const exception& e) {
        printf("caught: %s\n", e.what());
    }

    return 0;
}
