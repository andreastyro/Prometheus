#include "ml/tensor.hpp"
#include "ml/utils/testing.hpp"
#include "ml/nn/rnn/lstm.hpp"
#include <stdio.h>

using namespace std;

int main() {
    // === Shape test ===
    // input: [seq_len=5, batch=2, input_size=3], hidden_size=4
    // expected output: [5, 2, 4], h_n/c_n: [2, 4]
    printf("=== LSTM shape test ===\n");
    auto x = make_shared<Tensor>(Shape{5, 2, 3});
    for (int i = 0; i < x->num_el(); i++) x->data()[i] = 0.1f;

    LSTM lstm(3, 4);
    auto [out, hn, cn] = lstm.forward(x, nullptr, nullptr);
    printf("input:  [5, 2, 3]\n");
    CHECK_NEAR("output dim 0", (double)(out->shape()[0]), 5, 0.0);
CHECK_NEAR("output dim 1", (double)(out->shape()[1]), 2, 0.0);
CHECK_NEAR("output dim 2", (double)(out->shape()[2]), 4, 0.0);
    CHECK_NEAR("h_n dim 0", (double)(hn->shape()[0]), 2, 0.0);
CHECK_NEAR("h_n dim 1", (double)(hn->shape()[1]), 4, 0.0);
    CHECK_NEAR("c_n dim 0", (double)(cn->shape()[0]), 2, 0.0);
CHECK_NEAR("c_n dim 1", (double)(cn->shape()[1]), 4, 0.0);

    // === h_t is bounded by tanh(c) * sigmoid = (-1, 1) ===
    printf("\noutput values (expected in range (-1, 1)):\n");
    float min_val = out->data()[0], max_val = out->data()[0];
    for (float v : out->values()) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    printf("min: %.4f, max: %.4f\n", min_val, max_val);

    // === Custom h0 / c0 ===
    printf("\n=== LSTM with custom h0 / c0 ===\n");
    auto h0 = Tensor::zeros({2, 4});
    auto c0 = Tensor::zeros({2, 4});
    auto [out2, hn2, cn2] = lstm.forward(x, h0, c0);
    CHECK_NEAR("output dim 0", (double)(out2->shape()[0]), 5, 0.0);
CHECK_NEAR("output dim 1", (double)(out2->shape()[1]), 2, 0.0);
CHECK_NEAR("output dim 2", (double)(out2->shape()[2]), 4, 0.0);

    // === Module interface ===
    printf("\n=== LSTM as Module ===\n");
    auto out3 = lstm.forward(x);
    CHECK_NEAR("output dim 0", (double)(out3->shape()[0]), 5, 0.0);
CHECK_NEAR("output dim 1", (double)(out3->shape()[1]), 2, 0.0);
CHECK_NEAR("output dim 2", (double)(out3->shape()[2]), 4, 0.0);

    // === Parameter count ===
    printf("\n=== LSTM parameters ===\n");
    printf("param count: %zu (expected 12)\n", lstm.parameters().size());

    return test_report();
}
