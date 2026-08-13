#include "ml/nn/nlp/embedding.hpp"
#include "ml/autograd.hpp"
#include <stdexcept>
#include <string>

using namespace std;

Embedding::Embedding(int vocab_size, int embed_dim)
    : vocab_size(vocab_size), embed_dim(embed_dim) {
    weight = Tensor::randn({vocab_size, embed_dim});
    weight->requires_grad = true;
}

TensorPtr Embedding::forward(TensorPtr input) {
    int batch   = input->shape[0];
    int seq_len = input->shape[1];

    auto output = make_shared<Tensor>(vector<int>{batch, seq_len, embed_dim});

    vector<int> token_ids(batch * seq_len);
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq_len; t++) {
            int id = static_cast<int>(input->data[b * seq_len + t]);
            if (id < 0 || id >= vocab_size)
                throw runtime_error("Embedding: token id " + to_string(id)
                                    + " out of range [0, " + to_string(vocab_size) + ")");
            token_ids[b * seq_len + t] = id;

            int out_off = (b * seq_len + t) * embed_dim;
            int w_off   = id * embed_dim;
            for (int d = 0; d < embed_dim; d++)
                output->data[out_off + d] = weight->data[w_off + d];
        }
    }

    // Backward: scatter-add output->grad rows into the accessed weight rows.
    if (weight->requires_grad) {
        auto w = weight;
        auto node = make_node(output, {w});
        node->backward_fn = [w, output, token_ids, batch, seq_len, embed_dim=embed_dim]() {
            for (int b = 0; b < batch; b++) {
                for (int t = 0; t < seq_len; t++) {
                    int id      = token_ids[b * seq_len + t];
                    int out_off = (b * seq_len + t) * embed_dim;
                    int w_off   = id * embed_dim;
                    for (int d = 0; d < embed_dim; d++)
                        w->grad[w_off + d] += output->grad[out_off + d];
                }
            }
        };
    }

    return output;
}

vector<TensorPtr> Embedding::parameters() {
    return {weight};
}
