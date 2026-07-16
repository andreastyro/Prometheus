#include "ml/nn/lstm.hpp"
#include "ml/ops.hpp"

using namespace std;

LSTM::LSTM(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size) {

    W_xi = Tensor::randn({input_size, hidden_size}); W_xi->requires_grad = true;
    W_xf = Tensor::randn({input_size, hidden_size}); W_xf->requires_grad = true;
    W_xo = Tensor::randn({input_size, hidden_size}); W_xo->requires_grad = true;
    W_xg = Tensor::randn({input_size, hidden_size}); W_xg->requires_grad = true;

    W_hi = Tensor::randn({hidden_size, hidden_size}); W_hi->requires_grad = true;
    W_hf = Tensor::randn({hidden_size, hidden_size}); W_hf->requires_grad = true;
    W_ho = Tensor::randn({hidden_size, hidden_size}); W_ho->requires_grad = true;
    W_hg = Tensor::randn({hidden_size, hidden_size}); W_hg->requires_grad = true;

    b_i = Tensor::zeros({hidden_size}); b_i->requires_grad = true;
    b_f = Tensor::zeros({hidden_size}); b_f->requires_grad = true;
    b_o = Tensor::zeros({hidden_size}); b_o->requires_grad = true;
    b_g = Tensor::zeros({hidden_size}); b_g->requires_grad = true;
}

tuple<TensorPtr, TensorPtr, TensorPtr> LSTM::forward(TensorPtr input, TensorPtr h0, TensorPtr c0) {
    int seq_len  = input->shape[0];
    int batch    = input->shape[1];
    int input_sz = input->shape[2];

    TensorPtr h = h0 ? h0 : Tensor::zeros({batch, hidden_size});
    TensorPtr c = c0 ? c0 : Tensor::zeros({batch, hidden_size});

    auto output = make_shared<Tensor>(vector<int>{seq_len, batch, hidden_size});

    for (int t = 0; t < seq_len; t++) {
        // slice x_t: [batch, input_size]
        auto x_t = make_shared<Tensor>(vector<int>{batch, input_sz});
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < input_sz; i++)
                x_t->data[b * input_sz + i] = input->data[t * (batch * input_sz) + b * input_sz + i];

        // i_t = sigmoid(x_t @ W_xi + h @ W_hi + b_i)
        auto i_t = sigmoid(broadcast_add(add(matmul(x_t, W_xi), matmul(h, W_hi)), b_i));
        // f_t = sigmoid(x_t @ W_xf + h @ W_hf + b_f)
        auto f_t = sigmoid(broadcast_add(add(matmul(x_t, W_xf), matmul(h, W_hf)), b_f));
        // o_t = sigmoid(x_t @ W_xo + h @ W_ho + b_o)
        auto o_t = sigmoid(broadcast_add(add(matmul(x_t, W_xo), matmul(h, W_ho)), b_o));
        // g_t = tanh(x_t @ W_xg + h @ W_hg + b_g)
        auto g_t = tanh_op(broadcast_add(add(matmul(x_t, W_xg), matmul(h, W_hg)), b_g));

        // c_t = f_t * c + i_t * g_t
        c = add(multiply(f_t, c), multiply(i_t, g_t));
        // h_t = o_t * tanh(c_t)
        h = multiply(o_t, tanh_op(c));

        // store h into output[t]
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < hidden_size; i++)
                output->data[t * (batch * hidden_size) + b * hidden_size + i] = h->data[b * hidden_size + i];
    }

    return {output, h, c};
}

TensorPtr LSTM::forward(TensorPtr input) {
    return get<0>(forward(input, nullptr, nullptr));
}

vector<TensorPtr> LSTM::parameters() {
    return {W_xi, W_xf, W_xo, W_xg,
            W_hi, W_hf, W_ho, W_hg,
            b_i, b_f, b_o, b_g};
}
