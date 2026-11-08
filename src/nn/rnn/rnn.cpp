#include "ml/nn/rnn/rnn.hpp"
#include "ml/ops.hpp"
#include "ml/nn/rnn/sequence_ops.hpp"

using namespace std;

RNN::RNN(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size) {

    W_x  = Tensor::randn({input_size, hidden_size});
    W_h  = Tensor::randn({hidden_size, hidden_size});
    bias = Tensor::zeros({hidden_size});

    W_x->set_requires_grad(true);
    W_h->set_requires_grad(true);
    bias->set_requires_grad(true);
}

pair<TensorPtr, TensorPtr> RNN::forward(TensorPtr input, TensorPtr h0){
    int seq_len   = input->shape()[0];
    int batch     = input->shape()[1];
    int input_sz  = input->shape()[2];

    // initialize hidden state to zeros if not provided
    TensorPtr h = h0 ? h0 : Tensor::zeros({batch, hidden_size});

    // collected per timestep, then stacked differentiably at the end
    vector<TensorPtr> steps;
    steps.reserve(seq_len);

    for (int t = 0; t < seq_len; t++){

        // slice x_t: [batch, input_size] from input[t]
        auto x_t = rnn_slice_step(input, t);

        // h_t = tanh(x_t @ W_x + h_{t-1} @ W_h + bias)
        auto h_t = tanh_op(broadcast_add(add(matmul(x_t, W_x), matmul(h, W_h)), bias));

        steps.push_back(h_t);

        h = h_t;
    }

    auto output = rnn_stack_steps(steps);

    return {output, h};
}

TensorPtr RNN::forward(TensorPtr input){
    return forward(input, nullptr).first;
}



vector<TensorPtr> RNN::parameters(){
    return {W_x, W_h, bias};
}
