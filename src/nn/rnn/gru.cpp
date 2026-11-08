#include "ml/nn/rnn/gru.hpp"
#include "ml/ops.hpp"
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

pair<TensorPtr, TensorPtr> GRU::forward(TensorPtr input, TensorPtr h0) {
    int seq_len  = input->shape()[0];
    int batch    = input->shape()[1];
    int input_sz = input->shape()[2];

    TensorPtr h = h0 ? h0 : Tensor::zeros({batch, hidden_size});

    // collected per timestep, then stacked differentiably at the end
    vector<TensorPtr> steps;
    steps.reserve(seq_len);

    for (int t = 0; t < seq_len; t++) {
        // slice x_t: [batch, input_size]
        auto x_t = rnn_slice_step(input, t);

        // r_t = sigmoid(x_t @ W_xr + h @ W_hr + b_r)
        auto r_t = sigmoid(add(add(matmul(x_t, W_xr), matmul(h, W_hr)), b_r));
        // z_t = sigmoid(x_t @ W_xz + h @ W_hz + b_z)
        auto z_t = sigmoid(add(add(matmul(x_t, W_xz), matmul(h, W_hz)), b_z));
        // n_t = tanh(x_t @ W_xn + (r_t * h) @ W_hn + b_n)
        auto n_t = tanh_op(add(add(matmul(x_t, W_xn), matmul(multiply(r_t, h), W_hn)), b_n));
        // h_t = (1 - z_t) * n_t + z_t * h
        h = add(multiply(subtract(1.0f, z_t), n_t), multiply(z_t, h));

        steps.push_back(h);
    }

    auto output = rnn_stack_steps(steps);

    return {output, h};
}

TensorPtr GRU::forward(TensorPtr input) {
    return forward(input, nullptr).first;
}

vector<TensorPtr> GRU::parameters() {
    return {W_xr, W_xz, W_xn,
            W_hr, W_hz, W_hn,
            b_r, b_z, b_n};
}
