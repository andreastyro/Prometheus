#include "ml/tensor.hpp"
#include "ml/utils/testing.hpp"
#include "ml/nn/nlp/layernorm.hpp"
#include <stdio.h>
#include <cmath>

using namespace std;

int main() {
    // === Shape test ===
    // input: [2, 4] — 2 tokens, 4 features each
    printf("=== LayerNorm shape test ===\n");
    auto x = make_shared<Tensor>(Shape{2, 4});
    x->set_data({2.0f, 0.0f, 4.0f, 2.0f,
               1.0f, 3.0f, 1.0f, 3.0f});

    LayerNorm ln(4);
    auto out = ln.forward(x);
    printf("input:  [2, 4]\n");
    CHECK_NEAR("output dim 0", (double)(out->shape()[0]), 2, 0.0);
CHECK_NEAR("output dim 1", (double)(out->shape()[1]), 4, 0.0);

    // === After normalization each row should have mean~0, std~1 ===
    printf("\n=== Row 0 mean and std (expected ~0 and ~1) ===\n");
    float mean = 0.0f;
    for (int i = 0; i < 4; i++) mean += out->data()[i];
    mean /= 4;
    float var = 0.0f;
    for (int i = 0; i < 4; i++) var += (out->data()[i] - mean) * (out->data()[i] - mean);
    var /= 4;
    CHECK_NEAR("mean", (double)(mean), 0.0, 5e-2);
    CHECK_NEAR("std", (double)(sqrtf(var)), 1.0, 5e-2);

    // === gamma=1 beta=0 by default so output values check out ===
    printf("\nrow 0 values: ");
    for (int i = 0; i < 4; i++) printf("%.4f ", out->data()[i]);
    printf("\n");

    // === Works on 3D input [batch, seq, features] ===
    printf("\n=== 3D input [2, 3, 4] ===\n");
    auto x3 = make_shared<Tensor>(Shape{2, 3, 4});
    for (int i = 0; i < x3->num_el(); i++) x3->data()[i] = (float)i;
    auto out3 = ln.forward(x3);
    CHECK_NEAR("output dim 0", (double)(out3->shape()[0]), 2, 0.0);
CHECK_NEAR("output dim 1", (double)(out3->shape()[1]), 3, 0.0);
CHECK_NEAR("output dim 2", (double)(out3->shape()[2]), 4, 0.0);

    // === Parameters ===
    printf("\n=== Parameters ===\n");
    printf("param count: %zu (expected 2)\n", ln.parameters().size());

    return test_report();
}
