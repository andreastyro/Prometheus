#include "ml/tensor.hpp"
#include "ml/utils/testing.hpp"
#include "ml/nn/rnn/rnn.hpp"
#include <stdio.h>

using namespace std;

int main(){

    // === Shape test ===
    // input: [seq_len=5, batch=2, input_size=3]
    // hidden_size=4
    // expected output: [5, 2, 4], h_n: [2, 4]
    printf("=== RNN shape test ===\n");
    auto x1 = make_shared<Tensor>(Shape{5, 2, 3});
    for (int i = 0; i < x1->num_el(); i++) x1->data()[i] = 0.1f;

    RNN rnn1(3, 4);
    auto [out1, hn1] = rnn1.forward(x1, nullptr);
    printf("input:  [5, 2, 3]\n");
    CHECK_NEAR("output dim 0", (double)(out1->shape()[0]), 5, 0.0);
CHECK_NEAR("output dim 1", (double)(out1->shape()[1]), 2, 0.0);
CHECK_NEAR("output dim 2", (double)(out1->shape()[2]), 4, 0.0);
    CHECK_NEAR("h_n dim 0", (double)(hn1->shape()[0]), 2, 0.0);
CHECK_NEAR("h_n dim 1", (double)(hn1->shape()[1]), 4, 0.0);

    // === Values are bounded by tanh ===
    printf("\noutput values (expected in range [-1, 1]):\n");
    float min_val = out1->data()[0];
    float max_val = out1->data()[0];
    for (float v : out1->values()){
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    printf("min: %.4f, max: %.4f\n", min_val, max_val);

    // === Custom h0 test ===
    printf("\n=== RNN with custom h0 ===\n");
    auto h0 = Tensor::ones({2, 4});
    auto [out2, hn2] = rnn1.forward(x1, h0);
    CHECK_NEAR("output dim 0", (double)(out2->shape()[0]), 5, 0.0);
CHECK_NEAR("output dim 1", (double)(out2->shape()[1]), 2, 0.0);
CHECK_NEAR("output dim 2", (double)(out2->shape()[2]), 4, 0.0);

    // === Module interface (single arg) ===
    printf("\n=== RNN as Module ===\n");
    auto out3 = rnn1.forward(x1);
    CHECK_NEAR("output dim 0", (double)(out3->shape()[0]), 5, 0.0);
CHECK_NEAR("output dim 1", (double)(out3->shape()[1]), 2, 0.0);
CHECK_NEAR("output dim 2", (double)(out3->shape()[2]), 4, 0.0);

    return test_report();
}
