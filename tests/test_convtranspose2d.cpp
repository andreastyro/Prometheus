#include "ml/tensor.hpp"
#include "ml/nn/convtranspose2d.hpp"
#include <stdio.h>

using namespace std;

int main(){

    // === Shape test: stride=2 doubles spatial size ===
    // input: [1, 1, 4, 4], kernel=2, stride=2
    // h_out = (4-1)*2 - 0 + 2 = 8
    // expected output: [1, 1, 8, 8]
    printf("=== ConvTranspose2D shape test (stride=2) ===\n");
    auto x1 = make_shared<Tensor>(vector<int>{1, 1, 4, 4});
    for (int i = 0; i < x1->num_el(); i++) x1->data[i] = 1.0f;

    ConvTranspose2D ct1(1, 1, 2, 2);
    auto out1 = ct1.forward(x1);
    printf("input:  [1, 1, 4, 4]\n");
    printf("output: [%d, %d, %d, %d] (expected [1, 1, 8, 8])\n",
        out1->shape[0], out1->shape[1], out1->shape[2], out1->shape[3]);

    // === Shape test: stride=1 ===
    // input: [1, 4, 6, 6], kernel=2, stride=1
    // h_out = (6-1)*1 + 2 = 7
    // expected output: [1, 8, 7, 7]
    printf("\n=== ConvTranspose2D shape test (stride=1) ===\n");
    auto x2 = make_shared<Tensor>(vector<int>{1, 4, 6, 6});
    ConvTranspose2D ct2(4, 8, 2, 1);
    auto out2 = ct2.forward(x2);
    printf("input:  [1, 4, 6, 6]\n");
    printf("output: [%d, %d, %d, %d] (expected [1, 8, 7, 7])\n",
        out2->shape[0], out2->shape[1], out2->shape[2], out2->shape[3]);

    // === Backward test ===
    printf("\n=== ConvTranspose2D backward test ===\n");
    auto x3 = make_shared<Tensor>(vector<int>{1, 1, 2, 2});
    for (int i = 0; i < x3->num_el(); i++) x3->data[i] = 1.0f;
    x3->requires_grad = true;

    ConvTranspose2D ct3(1, 1, 2, 2);
    auto out3 = ct3.forward(x3);
    for (int i = 0; i < out3->num_el(); i++) out3->grad[i] = 1.0f;
    out3->grad_fn->backward_fn();

    printf("weights->grad (expected non-zero):\n");
    for (int i = 0; i < ct3.weights->num_el(); i++)
        printf("  [%d]: %.4f\n", i, ct3.weights->grad[i]);
    printf("input->grad (expected non-zero):\n");
    for (int i = 0; i < x3->num_el(); i++)
        printf("  [%d]: %.4f\n", i, x3->grad[i]);

    return 0;
}
