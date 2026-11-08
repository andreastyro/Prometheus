#include "ml/nn/rnn/gru.hpp"
#include "ml/ops.hpp"
#include "ml/nn/rnn/fused_rnn.hpp"
#include "ml/nn/rnn/sequence_ops.hpp"

using namespace std;

GRU::GRU(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size) {

    W_xr = Tensor::randn({input_size, hidden_size}); W_xr->set_requires_grad(true);
    W_xz = Tensor::randn({input_size, hidden_size}); W_xz->set_requires_grad(true);
    W_xn = Tensor::randn({input_size, hidden_size}); W_xn->set_requires_grad(true);

    W_hr = Tensor::randn({hidden_size, hidden_size}); W_hr->set_requires_grad(true);
    W_hz = Tensor::randn({hidden_size, hidden_size}); W_hz->set_requires_grad(true);
    W_hn = Tensor::randn({hidden_size, hidden_size}); W_hn->set_requires_grad(true);

    b_r = Tensor::zeros({hidden_size}); b_r->set_requires_grad(true);
    b_z = Tensor::zeros({hidden_size}); b_z->set_requires_grad(true);
    b_n = Tensor::zeros({hidden_size}); b_n->set_requires_grad(true);
}

// The whole sequence in one fused kernel; see fused_rnn.hpp.
pair<TensorPtr, TensorPtr> GRU::forward(TensorPtr input, TensorPtr h0) {
    auto r = rnn_kernel::run(rnn_kernel::Cell::GRU, input, h0, nullptr, hidden_size,
                             {W_xr, W_xz, W_xn}, {W_hr, W_hz, W_hn}, {b_r, b_z, b_n});
    return {r.output, r.h_n};
}

TensorPtr GRU::forward(TensorPtr input) {
    return forward(input, nullptr).first;
}

vector<TensorPtr> GRU::parameters() {
    return {W_xr, W_xz, W_xn,
            W_hr, W_hz, W_hn,
            b_r, b_z, b_n};
}
