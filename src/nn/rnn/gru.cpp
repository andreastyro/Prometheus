#include "ml/nn/rnn/gru.hpp"
#include "ml/ops.hpp"

using namespace std;

GRU::GRU(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size) {

    W_xr = Tensor::randn({input_size, hidden_size}); W_xr->requires_grad = true;
    W_xz = Tensor::randn({input_size, hidden_size}); W_xz->requires_grad = true;
    W_xn = Tensor::randn({input_size, hidden_size}); W_xn->requires_grad = true;

    W_hr = Tensor::randn({hidden_size, hidden_size}); W_hr->requires_grad = true;
    W_hz = Tensor::randn({hidden_size, hidden_size}); W_hz->requires_grad = true;
    W_hn = Tensor::randn({hidden_size, hidden_size}); W_hn->requires_grad = true;

    b_r = Tensor::zeros({hidden_size}); b_r->requires_grad = true;
    b_z = Tensor::zeros({hidden_size}); b_z->requires_grad = true;
    b_n = Tensor::zeros({hidden_size}); b_n->requires_grad = true;
}

pair<TensorPtr, TensorPtr> GRU::forward(TensorPtr input, TensorPtr h0) {
    int seq_len  = input->shape[0];
    int batch    = input->shape[1];
    int input_sz = input->shape[2];

    TensorPtr h = h0 ? h0 : Tensor::zeros({batch, hidden_size});

    auto output = make_shared<Tensor>(vector<int>{seq_len, batch, hidden_size});

    for (int t = 0; t < seq_len; t++) {
        // slice x_t: [batch, input_size]
        auto x_t = make_shared<Tensor>(vector<int>{batch, input_sz});
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < input_sz; i++)
                x_t->data[b * input_sz + i] = input->data[t * (batch * input_sz) + b * input_sz + i];

        // r_t = sigmoid(x_t @ W_xr + h @ W_hr + b_r)
        auto r_t = sigmoid(broadcast_add(add(matmul(x_t, W_xr), matmul(h, W_hr)), b_r));
        // z_t = sigmoid(x_t @ W_xz + h @ W_hz + b_z)
        auto z_t = sigmoid(broadcast_add(add(matmul(x_t, W_xz), matmul(h, W_hz)), b_z));
        // n_t = tanh(x_t @ W_xn + (r_t * h) @ W_hn + b_n)
        auto n_t = tanh_op(broadcast_add(add(matmul(x_t, W_xn), matmul(multiply(r_t, h), W_hn)), b_n));
        // h_t = (1 - z_t) * n_t + z_t * h
        h = add(multiply(subtract(1.0f, z_t), n_t), multiply(z_t, h));

        // store h into output[t]
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < hidden_size; i++)
                output->data[t * (batch * hidden_size) + b * hidden_size + i] = h->data[b * hidden_size + i];
    }

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
