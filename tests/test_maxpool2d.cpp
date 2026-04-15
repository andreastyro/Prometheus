#include "ml/tensor.hpp"
#include "ml/nn/maxpool2d.hpp"
#include "ml/nn/avgpool2d.hpp"
#include "ml/nn/conv2d.hpp"
#include "ml/nn/flatten.hpp"
#include <stdio.h>

using namespace std;

int main(){

    // === Shape test ===
    // input: [1, 1, 4, 4], kernel=2, stride=2
    // expected output: [1, 1, 2, 2]
    printf("=== MaxPool2D shape test ===\n");
    auto x1 = make_shared<Tensor>(
        vector<int>{1, 1, 4, 4},
        vector<float>{
            1, 3, 2, 4,
            5, 6, 1, 2,
            3, 1, 4, 7,
            2, 8, 5, 3
        }
    );

    MaxPool2D pool1(2);
    auto out1 = pool1.forward(x1);
    printf("input:  [1, 1, 4, 4]\n");
    printf("output: [%d, %d, %d, %d] (expected [1, 1, 2, 2])\n",
        out1->shape[0], out1->shape[1], out1->shape[2], out1->shape[3]);
    printf("values: %.0f %.0f %.0f %.0f (expected 6 4 8 7)\n",
        out1->data[0], out1->data[1], out1->data[2], out1->data[3]);

    // === Batch + channels test ===
    // input: [2, 3, 4, 4], kernel=2, stride=2
    // expected output: [2, 3, 2, 2]
    printf("\n=== MaxPool2D batch test ===\n");
    auto x2 = make_shared<Tensor>(vector<int>{2, 3, 4, 4});
    for (int i = 0; i < x2->num_el(); i++) x2->data[i] = (float)(i + 1);

    MaxPool2D pool2(2);
    auto out2 = pool2.forward(x2);
    printf("input:  [2, 3, 4, 4]\n");
    printf("output: [%d, %d, %d, %d] (expected [2, 3, 2, 2])\n",
        out2->shape[0], out2->shape[1], out2->shape[2], out2->shape[3]);

    // === Backward test ===
    printf("\n=== MaxPool2D backward test ===\n");
    auto x3 = make_shared<Tensor>(
        vector<int>{1, 1, 4, 4},
        vector<float>{
            1, 3, 2, 4,
            5, 6, 1, 2,
            3, 1, 4, 7,
            2, 8, 5, 3
        }
    );
    x3->requires_grad = true;

    MaxPool2D pool3(2);
    auto out3 = pool3.forward(x3);

    // set all output grads to 1
    for (int i = 0; i < out3->num_el(); i++) out3->grad[i] = 1.0f;
    out3->grad_fn->backward_fn();

    printf("input->grad (max positions get 1, rest get 0):\n");
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++)
            printf("%.0f ", x3->grad[i * 4 + j]);
        printf("\n");
    }
    printf("(expected 6,4,8,7 positions to have grad=1)\n");

    // === Conv2D + MaxPool2D + Flatten ===
    printf("\n=== Conv2D + MaxPool2D + Flatten ===\n");
    auto x4 = make_shared<Tensor>(vector<int>{1, 1, 8, 8});
    Conv2D conv(1, 4, 3);
    MaxPool2D pool4(2);
    Flatten flatten;

    auto conv_out = conv.forward(x4);       // [1, 4, 6, 6]
    auto pool_out = pool4.forward(conv_out); // [1, 4, 3, 3]
    auto flat_out = flatten.forward(pool_out); // [1, 36]

    printf("input:   [1, 1, 8, 8]\n");
    printf("conv:    [%d, %d, %d, %d]\n", conv_out->shape[0], conv_out->shape[1], conv_out->shape[2], conv_out->shape[3]);
    printf("pool:    [%d, %d, %d, %d]\n", pool_out->shape[0], pool_out->shape[1], pool_out->shape[2], pool_out->shape[3]);
    printf("flatten: [%d, %d] (expected [1, 36])\n", flat_out->shape[0], flat_out->shape[1]);

    // === AvgPool2D ===
    printf("\n=== AvgPool2D shape test ===\n");
    auto x5 = make_shared<Tensor>(
        vector<int>{1, 1, 4, 4},
        vector<float>{
            1, 3, 2, 4,
            5, 6, 1, 2,
            3, 1, 4, 7,
            2, 8, 5, 3
        }
    );

    AvgPool2D avgpool(2);
    auto out5 = avgpool.forward(x5);
    printf("input:  [1, 1, 4, 4]\n");
    printf("output: [%d, %d, %d, %d] (expected [1, 1, 2, 2])\n",
        out5->shape[0], out5->shape[1], out5->shape[2], out5->shape[3]);
    // top-left window: (1+3+5+6)/4=3.75, top-right: (2+4+1+2)/4=2.25
    // bot-left: (3+1+2+8)/4=3.5,         bot-right: (4+7+5+3)/4=4.75
    printf("values: %.2f %.2f %.2f %.2f (expected 3.75 2.25 3.50 4.75)\n",
        out5->data[0], out5->data[1], out5->data[2], out5->data[3]);

    // === AvgPool2D backward ===
    printf("\n=== AvgPool2D backward test ===\n");
    auto x6 = make_shared<Tensor>(
        vector<int>{1, 1, 4, 4},
        vector<float>{1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16}
    );
    x6->requires_grad = true;

    AvgPool2D avgpool2(2);
    auto out6 = avgpool2.forward(x6);
    for (int i = 0; i < out6->num_el(); i++) out6->grad[i] = 1.0f;
    out6->grad_fn->backward_fn();

    printf("input->grad (expected all 0.25):\n");
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++)
            printf("%.2f ", x6->grad[i * 4 + j]);
        printf("\n");
    }

    return 0;
}
