#include "ml/nn/rnn/lstm.hpp"
#include "ml/ops.hpp"
#include "ml/nn/rnn/fused_rnn.hpp"
#include "ml/nn/rnn/sequence_ops.hpp"

using namespace std;

LSTM::LSTM(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size) {

    W_xi = rnn_kernel::init_param({input_size, hidden_size}, hidden_size); W_xi->set_requires_grad(true);
    W_xf = rnn_kernel::init_param({input_size, hidden_size}, hidden_size); W_xf->set_requires_grad(true);
    W_xo = rnn_kernel::init_param({input_size, hidden_size}, hidden_size); W_xo->set_requires_grad(true);
    W_xg = rnn_kernel::init_param({input_size, hidden_size}, hidden_size); W_xg->set_requires_grad(true);

    W_hi = rnn_kernel::init_param({hidden_size, hidden_size}, hidden_size); W_hi->set_requires_grad(true);
    W_hf = rnn_kernel::init_param({hidden_size, hidden_size}, hidden_size); W_hf->set_requires_grad(true);
    W_ho = rnn_kernel::init_param({hidden_size, hidden_size}, hidden_size); W_ho->set_requires_grad(true);
    W_hg = rnn_kernel::init_param({hidden_size, hidden_size}, hidden_size); W_hg->set_requires_grad(true);

    b_i = rnn_kernel::init_param({hidden_size}, hidden_size); b_i->set_requires_grad(true);
    b_f = rnn_kernel::init_param({hidden_size}, hidden_size); b_f->set_requires_grad(true);
    b_o = rnn_kernel::init_param({hidden_size}, hidden_size); b_o->set_requires_grad(true);
    b_g = rnn_kernel::init_param({hidden_size}, hidden_size); b_g->set_requires_grad(true);
}

// The whole sequence in one fused kernel; see fused_rnn.hpp.
tuple<TensorPtr, TensorPtr, TensorPtr> LSTM::forward(TensorPtr input, TensorPtr h0, TensorPtr c0) {
    auto r = rnn_kernel::run(rnn_kernel::Cell::LSTM, input, h0, c0, hidden_size,
                             {W_xi, W_xf, W_xo, W_xg}, {W_hi, W_hf, W_ho, W_hg}, {b_i, b_f, b_o, b_g});
    return {r.output, r.h_n, r.c_n};
}

TensorPtr LSTM::forward(TensorPtr input) {
    return get<0>(forward(input, nullptr, nullptr));
}

vector<TensorPtr> LSTM::parameters() {
    return {W_xi, W_xf, W_xo, W_xg,
            W_hi, W_hf, W_ho, W_hg,
            b_i, b_f, b_o, b_g};
}
