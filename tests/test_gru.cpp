#include "ml/tensor.hpp"
#include "ml/utils/testing.hpp"
#include "ml/nn/rnn/gru.hpp"
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
    CHECK_NEAR("output dim 0", (double)(out->shape[0]), 5, 0.0);
CHECK_NEAR("output dim 1", (double)(out->shape[1]), 2, 0.0);
CHECK_NEAR("output dim 2", (double)(out->shape[2]), 4, 0.0);
    CHECK_NEAR("h_n dim 0", (double)(hn->shape[0]), 2, 0.0);
CHECK_NEAR("h_n dim 1", (double)(hn->shape[1]), 4, 0.0);

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
    CHECK_NEAR("output dim 0", (double)(out2->shape[0]), 5, 0.0);
CHECK_NEAR("output dim 1", (double)(out2->shape[1]), 2, 0.0);
CHECK_NEAR("output dim 2", (double)(out2->shape[2]), 4, 0.0);

    // === Module interface ===
    printf("\n=== GRU as Module ===\n");
    auto out3 = gru.forward(x);
    CHECK_NEAR("output dim 0", (double)(out3->shape[0]), 5, 0.0);
CHECK_NEAR("output dim 1", (double)(out3->shape[1]), 2, 0.0);
CHECK_NEAR("output dim 2", (double)(out3->shape[2]), 4, 0.0);

    // === Parameter count ===
    printf("\n=== GRU parameters ===\n");
    printf("param count: %zu (expected 9)\n", gru.parameters().size());

    return test_report();
}
