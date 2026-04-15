#include "ml/tensor.hpp"
#include "ml/nn/rnn.hpp"
#include <stdio.h>

using namespace std;

int main(){

    // === Shape test ===
    // input: [seq_len=5, batch=2, input_size=3]
    // hidden_size=4
    // expected output: [5, 2, 4], h_n: [2, 4]
    printf("=== RNN shape test ===\n");
    auto x1 = make_shared<Tensor>(vector<int>{5, 2, 3});
    for (int i = 0; i < x1->num_el(); i++) x1->data[i] = 0.1f;

    RNN rnn1(3, 4);
    auto [out1, hn1] = rnn1.forward(x1, nullptr);
    printf("input:  [5, 2, 3]\n");
    printf("output: [%d, %d, %d] (expected [5, 2, 4])\n",
        out1->shape[0], out1->shape[1], out1->shape[2]);
    printf("h_n:    [%d, %d] (expected [2, 4])\n",
        hn1->shape[0], hn1->shape[1]);

    // === Values are bounded by tanh ===
    printf("\noutput values (expected in range [-1, 1]):\n");
    float min_val = out1->data[0];
    float max_val = out1->data[0];
    for (float v : out1->data){
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    printf("min: %.4f, max: %.4f\n", min_val, max_val);

    // === Custom h0 test ===
    printf("\n=== RNN with custom h0 ===\n");
    auto h0 = Tensor::ones({2, 4});
    auto [out2, hn2] = rnn1.forward(x1, h0);
    printf("output: [%d, %d, %d] (expected [5, 2, 4])\n",
        out2->shape[0], out2->shape[1], out2->shape[2]);

    // === Module interface (single arg) ===
    printf("\n=== RNN as Module ===\n");
    auto out3 = rnn1.forward(x1);
    printf("output: [%d, %d, %d] (expected [5, 2, 4])\n",
        out3->shape[0], out3->shape[1], out3->shape[2]);

    return 0;
}
