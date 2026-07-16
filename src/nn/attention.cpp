#include "ml/nn/attention.hpp"
#include "ml/ops.hpp"
#include <cmath>
#include <stdexcept>

using namespace std;

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads)
    : embed_dim(embed_dim), num_heads(num_heads), head_dim(embed_dim / num_heads) {

    if (embed_dim % num_heads != 0)
        throw runtime_error("MultiHeadAttention: embed_dim must be divisible by num_heads");

    W_q = Tensor::randn({embed_dim, embed_dim}); W_q->requires_grad = true;
    W_k = Tensor::randn({embed_dim, embed_dim}); W_k->requires_grad = true;
    W_v = Tensor::randn({embed_dim, embed_dim}); W_v->requires_grad = true;
    W_o = Tensor::randn({embed_dim, embed_dim}); W_o->requires_grad = true;
}

TensorPtr MultiHeadAttention::forward(TensorPtr input) {
    int seq_len = input->shape[0];
    float scale = 1.0f / std::sqrt((float)head_dim);

    // project input to Q, K, V — each [seq_len, embed_dim]
    auto Q = matmul(input, W_q);
    auto K = matmul(input, W_k);
    auto V = matmul(input, W_v);

    // output accumulator [seq_len, embed_dim]
    auto concat = make_shared<Tensor>(vector<int>{seq_len, embed_dim});

    for (int h = 0; h < num_heads; h++) {
        int start = h * head_dim;

        // slice Q_h, K_h, V_h: [seq_len, head_dim]
        auto Q_h = make_shared<Tensor>(vector<int>{seq_len, head_dim});
        auto K_h = make_shared<Tensor>(vector<int>{seq_len, head_dim});
        auto V_h = make_shared<Tensor>(vector<int>{seq_len, head_dim});

        for (int t = 0; t < seq_len; t++)
            for (int d = 0; d < head_dim; d++) {
                Q_h->data[t * head_dim + d] = Q->data[t * embed_dim + start + d];
                K_h->data[t * head_dim + d] = K->data[t * embed_dim + start + d];
                V_h->data[t * head_dim + d] = V->data[t * embed_dim + start + d];
            }

        // scores = Q_h @ K_h.T / sqrt(head_dim) — [seq_len, seq_len]
        auto K_h_T = K_h->transpose();
        auto scores = matmul(Q_h, K_h_T);
        for (float& v : scores->data) v *= scale;

        // softmax over each row — each token's attention weights over all tokens
        auto weights = make_shared<Tensor>(vector<int>{seq_len, seq_len});
        for (int t = 0; t < seq_len; t++) {
            // find max for numerical stability
            float max_val = scores->data[t * seq_len];
            for (int j = 1; j < seq_len; j++)
                if (scores->data[t * seq_len + j] > max_val)
                    max_val = scores->data[t * seq_len + j];

            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                weights->data[t * seq_len + j] = std::exp(scores->data[t * seq_len + j] - max_val);
                sum += weights->data[t * seq_len + j];
            }
            for (int j = 0; j < seq_len; j++)
                weights->data[t * seq_len + j] /= sum;
        }

        // head output = weights @ V_h — [seq_len, head_dim]
        auto head_out = matmul(weights, V_h);

        // write head output into the correct columns of concat
        for (int t = 0; t < seq_len; t++)
            for (int d = 0; d < head_dim; d++)
                concat->data[t * embed_dim + start + d] = head_out->data[t * head_dim + d];
    }

    // final projection: concat @ W_o — [seq_len, embed_dim]
    return matmul(concat, W_o);
}

vector<TensorPtr> MultiHeadAttention::parameters() {
    return {W_q, W_k, W_v, W_o};
}
