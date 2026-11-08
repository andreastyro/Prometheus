#include "ml/nn/rnn/rnn.hpp"
#include "ml/ops.hpp"
#include "ml/nn/rnn/fused_rnn.hpp"
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

// The whole sequence in one fused kernel; see fused_rnn.hpp.
pair<TensorPtr, TensorPtr> RNN::forward(TensorPtr input, TensorPtr h0){
    auto r = rnn_kernel::run(rnn_kernel::Cell::RNN, input, h0, nullptr, hidden_size,
                             {W_x}, {W_h}, {bias});
    return {r.output, r.h_n};
}

TensorPtr RNN::forward(TensorPtr input){
    return forward(input, nullptr).first;
}



vector<TensorPtr> RNN::parameters(){
    return {W_x, W_h, bias};
}
