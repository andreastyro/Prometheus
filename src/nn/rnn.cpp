#include "ml/nn/rnn.hpp"
#include "ml/ops.hpp"

using namespace std;

RNN::RNN(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size) {

    W_x  = Tensor::randn({input_size, hidden_size});
    W_h  = Tensor::randn({hidden_size, hidden_size});
    bias = Tensor::zeros({hidden_size});

    W_x->requires_grad  = true;
    W_h->requires_grad  = true;
    bias->requires_grad = true;
}

pair<TensorPtr, TensorPtr> RNN::forward(TensorPtr input, TensorPtr h0){
    int seq_len   = input->shape[0];
    int batch     = input->shape[1];
    int input_sz  = input->shape[2];

    // initialize hidden state to zeros if not provided
    TensorPtr h = h0 ? h0 : Tensor::zeros({batch, hidden_size});

    auto output = make_shared<Tensor>(vector<int>{seq_len, batch, hidden_size});

    for (int t = 0; t < seq_len; t++){

        // slice x_t: [batch, input_size] from input[t]
        auto x_t = make_shared<Tensor>(vector<int>{batch, input_sz});
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < input_sz; i++)
                x_t->data[b * input_sz + i] = input->data[t * (batch * input_sz) + b * input_sz + i];

        // h_t = tanh(x_t @ W_x + h_{t-1} @ W_h + bias)
        auto h_t = tanh_op(broadcast_add(add(matmul(x_t, W_x), matmul(h, W_h)), bias));

        // store h_t into output[t]
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < hidden_size; i++)
                output->data[t * (batch * hidden_size) + b * hidden_size + i] = h_t->data[b * hidden_size + i];

        h = h_t;
    }

    return {output, h};
}

TensorPtr RNN::forward(TensorPtr input){
    return forward(input, nullptr).first;
}



vector<TensorPtr> RNN::parameters(){
    return {W_x, W_h, bias};
}
