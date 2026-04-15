#include "ml/tensor.hpp"
#include "ml/nn/conv2d.hpp"
#include "ml/nn/flatten.hpp"
#include "ml/autograd.hpp"
#include <stdio.h>

using namespace std;

int main(){

    // === Conv2D forward shape test ===
    // input: [1, 1, 5, 5] — 1 sample, 1 channel, 5x5
    // conv:  1 -> 2 filters, kernel=3, stride=1, padding=0
    // expected output: [1, 2, 3, 3]
    printf("=== Conv2D shape test (no padding) ===\n");
    auto x1 = make_shared<Tensor>(vector<int>{1, 1, 5, 5});
    for (int i = 0; i < x1->num_el(); i++) x1->data[i] = (float)(i + 1);

    Conv2D conv1(1, 2, 3);
    auto out1 = conv1.forward(x1);
    printf("input:  [1, 1, 5, 5]\n");
    printf("output: [%d, %d, %d, %d] (expected [1, 2, 3, 3])\n",
        out1->shape[0], out1->shape[1], out1->shape[2], out1->shape[3]);

    // === Conv2D with padding ===
    // input: [1, 1, 5, 5], kernel=3, padding=1
    // expected output: [1, 2, 5, 5] — same spatial size
    printf("\n=== Conv2D shape test (padding=1) ===\n");
    Conv2D conv2(1, 2, 3, 1, 1);
    auto out2 = conv2.forward(x1);
    printf("input:  [1, 1, 5, 5]\n");
    printf("output: [%d, %d, %d, %d] (expected [1, 2, 5, 5])\n",
        out2->shape[0], out2->shape[1], out2->shape[2], out2->shape[3]);

    // === Conv2D with stride=2 ===
    // input: [1, 1, 5, 5], kernel=3, stride=2
    // expected output: [1, 2, 2, 2]
    printf("\n=== Conv2D shape test (stride=2) ===\n");
    Conv2D conv3(1, 2, 3, 2, 0);
    auto out3 = conv3.forward(x1);
    printf("input:  [1, 1, 5, 5]\n");
    printf("output: [%d, %d, %d, %d] (expected [1, 2, 2, 2])\n",
        out3->shape[0], out3->shape[1], out3->shape[2], out3->shape[3]);

    // === Batch test ===
    // input: [4, 3, 8, 8] — 4 samples, 3 channels (RGB), 8x8
    // conv:  3 -> 16 filters, kernel=3, stride=1, padding=0
    // expected output: [4, 16, 6, 6]
    printf("\n=== Conv2D batch test ===\n");
    auto x2 = make_shared<Tensor>(vector<int>{4, 3, 8, 8});
    Conv2D conv4(3, 16, 3);
    auto out4 = conv4.forward(x2);
    printf("input:  [4, 3, 8, 8]\n");
    printf("output: [%d, %d, %d, %d] (expected [4, 16, 6, 6])\n",
        out4->shape[0], out4->shape[1], out4->shape[2], out4->shape[3]);

    // === Conv2D + Flatten ===
    // output of conv4 is [4, 16, 6, 6] -> flatten to [4, 576]
    printf("\n=== Conv2D + Flatten ===\n");
    Flatten flatten;
    auto flat = flatten.forward(out4);
    printf("after flatten: [%d, %d] (expected [4, 576])\n",
        flat->shape[0], flat->shape[1]);

    // === Backward test ===
    // input: [1, 1, 3, 3], 1 filter, kernel=2
    // run forward, set output grad to 1s, call backward
    // check that weights->grad and input->grad are non-zero
    printf("\n=== Conv2D backward test ===\n");
    auto x3 = make_shared<Tensor>(
        vector<int>{1, 1, 3, 3},
        vector<float>{1,2,3, 4,5,6, 7,8,9}
    );
    x3->requires_grad = true;

    Conv2D conv5(1, 1, 2);
    auto out5 = conv5.forward(x3);

    // set all output grads to 1
    for (int i = 0; i < out5->num_el(); i++) out5->grad[i] = 1.0f;

    // run backward
    out5->grad_fn->backward_fn();

    printf("weights->grad (expected non-zero):\n");
    for (int i = 0; i < conv5.weights->num_el(); i++)
        printf("  [%d]: %.4f\n", i, conv5.weights->grad[i]);

    printf("bias->grad (expected non-zero): %.4f\n", conv5.bias->grad[0]);

    printf("input->grad (expected non-zero):\n");
    for (int i = 0; i < x3->num_el(); i++)
        printf("  [%d]: %.4f\n", i, x3->grad[i]);

    return 0;
}
