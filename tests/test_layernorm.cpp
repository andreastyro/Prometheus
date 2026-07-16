#include "ml/tensor.hpp"
#include "ml/nn/layernorm.hpp"
#include <stdio.h>
#include <cmath>

using namespace std;

int main() {
    // === Shape test ===
    // input: [2, 4] — 2 tokens, 4 features each
    printf("=== LayerNorm shape test ===\n");
    auto x = make_shared<Tensor>(vector<int>{2, 4});
    x->data = {2.0f, 0.0f, 4.0f, 2.0f,
               1.0f, 3.0f, 1.0f, 3.0f};

    LayerNorm ln(4);
    auto out = ln.forward(x);
    printf("input:  [2, 4]\n");
    printf("output: [%d, %d] (expected [2, 4])\n", out->shape[0], out->shape[1]);

    // === After normalization each row should have mean~0, std~1 ===
    printf("\n=== Row 0 mean and std (expected ~0 and ~1) ===\n");
    float mean = 0.0f;
    for (int i = 0; i < 4; i++) mean += out->data[i];
    mean /= 4;
    float var = 0.0f;
    for (int i = 0; i < 4; i++) var += (out->data[i] - mean) * (out->data[i] - mean);
    var /= 4;
    printf("mean: %.4f (expected ~0.0)\n", mean);
    printf("std:  %.4f (expected ~1.0)\n", sqrtf(var));

    // === gamma=1 beta=0 by default so output values check out ===
    printf("\nrow 0 values: ");
    for (int i = 0; i < 4; i++) printf("%.4f ", out->data[i]);
    printf("\n");

    // === Works on 3D input [batch, seq, features] ===
    printf("\n=== 3D input [2, 3, 4] ===\n");
    auto x3 = make_shared<Tensor>(vector<int>{2, 3, 4});
    for (int i = 0; i < x3->num_el(); i++) x3->data[i] = (float)i;
    auto out3 = ln.forward(x3);
    printf("output: [%d, %d, %d] (expected [2, 3, 4])\n",
        out3->shape[0], out3->shape[1], out3->shape[2]);

    // === Parameters ===
    printf("\n=== Parameters ===\n");
    printf("param count: %zu (expected 2)\n", ln.parameters().size());

    return 0;
}
