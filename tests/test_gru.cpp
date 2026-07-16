#include "ml/tensor.hpp"
#include "ml/nn/gru.hpp"
#include <stdio.h>

using namespace std;

int main() {
    // === Shape test ===
    // input: [seq_len=5, batch=2, input_size=3], hidden_size=4
    // expected output: [5, 2, 4], h_n: [2, 4]
    printf("=== GRU shape test ===\n");
    auto x = make_shared<Tensor>(vector<int>{5, 2, 3});
    for (int i = 0; i < x->num_el(); i++) x->data[i] = 0.1f;

    GRU gru(3, 4);
    auto [out, hn] = gru.forward(x, nullptr);
    printf("input:  [5, 2, 3]\n");
    printf("output: [%d, %d, %d] (expected [5, 2, 4])\n",
        out->shape[0], out->shape[1], out->shape[2]);
    printf("h_n:    [%d, %d] (expected [2, 4])\n", hn->shape[0], hn->shape[1]);

    // === h_t bounded by tanh and linear interp of gates ===
    printf("\noutput values (expected in range (-1, 1)):\n");
    float min_val = out->data[0], max_val = out->data[0];
    for (float v : out->data) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    printf("min: %.4f, max: %.4f\n", min_val, max_val);

    // === Custom h0 ===
    printf("\n=== GRU with custom h0 ===\n");
    auto h0 = Tensor::zeros({2, 4});
    auto [out2, hn2] = gru.forward(x, h0);
    printf("output: [%d, %d, %d] (expected [5, 2, 4])\n",
        out2->shape[0], out2->shape[1], out2->shape[2]);

    // === Module interface ===
    printf("\n=== GRU as Module ===\n");
    auto out3 = gru.forward(x);
    printf("output: [%d, %d, %d] (expected [5, 2, 4])\n",
        out3->shape[0], out3->shape[1], out3->shape[2]);

    // === Parameter count ===
    printf("\n=== GRU parameters ===\n");
    printf("param count: %zu (expected 9)\n", gru.parameters().size());

    return 0;
}
