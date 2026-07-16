#include "ml/tensor.hpp"
#include "ml/nn/groupnorm.hpp"
#include <stdio.h>
#include <cmath>

using namespace std;

int main() {
    // input: [batch=2, channels=4, spatial=3]
    // 2 groups of 2 channels each
    printf("=== GroupNorm shape test ===\n");
    auto x = make_shared<Tensor>(vector<int>{2, 4, 3});
    for (int i = 0; i < x->num_el(); i++) x->data[i] = (float)(i + 1);

    GroupNorm gn(2, 4);
    auto out = gn.forward(x);
    printf("input:  [2, 4, 3]\n");
    printf("output: [%d, %d, %d] (expected [2, 4, 3])\n",
        out->shape[0], out->shape[1], out->shape[2]);

    // === Each group in each batch should have mean~0 ===
    // group 0 of batch 0: channels 0-1, 3 spatial = 6 values
    printf("\n=== Group 0 batch 0 mean (expected ~0) ===\n");
    float mean = 0.0f;
    for (int c = 0; c < 2; c++)
        for (int s = 0; s < 3; s++)
            mean += out->data[(0 * 4 + c) * 3 + s];
    mean /= 6;
    printf("mean: %.4f\n", mean);

    // === Invalid groups throws ===
    printf("\n=== Invalid num_groups throws ===\n");
    try {
        GroupNorm bad(3, 4);  // 4 not divisible by 3
        printf("no exception (unexpected)\n");
    } catch (const exception& e) {
        printf("caught: %s\n", e.what());
    }

    // === Parameters ===
    printf("\n=== Parameters ===\n");
    auto params = gn.parameters();
    printf("param count: %zu (expected 2)\n", params.size());
    printf("gamma shape: [%d] (expected [4])\n", params[0]->shape[0]);

    return 0;
}
