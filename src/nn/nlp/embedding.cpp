#include "ml/nn/nlp/embedding.hpp"
#include "ml/ops.hpp"
#include "ml/autograd.hpp"
#include <stdexcept>
#include <string>

using namespace std;

Embedding::Embedding(int vocab_size, int embed_dim)
    : vocab_size(vocab_size), embed_dim(embed_dim) {
    weight = Tensor::randn({vocab_size, embed_dim});
    weight->set_requires_grad(true);
}

TensorPtr Embedding::forward(TensorPtr input) {
    if (input->dim() != 1 && input->dim() != 2)
        throw runtime_error("Embedding: ids must be [batch, seq] or [seq]");
    const bool batched = input->dim() == 2;
    int batch   = batched ? (int)input->shape()[0] : 1;
    int seq_len = (int)input->shape().back();
    // exact for any dtype: int64 ids are not rounded through a float
    const vector<int64_t> ids = index_values(input, "Embedding");

    auto output = batched ? make_shared<Tensor>(Shape{batch, seq_len, embed_dim})
                          : make_shared<Tensor>(Shape{seq_len, embed_dim});

    vector<int> token_ids(batch * seq_len);
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq_len; t++) {
            int id = static_cast<int>(ids[b * seq_len + t]);
            if (id < 0 || id >= vocab_size)
                throw runtime_error("Embedding: token id " + to_string(id)
                                    + " out of range [0, " + to_string(vocab_size) + ")");
            token_ids[b * seq_len + t] = id;

            int out_off = (b * seq_len + t) * embed_dim;
            int w_off   = id * embed_dim;
            for (int d = 0; d < embed_dim; d++)
                output->data()[out_off + d] = weight->data()[w_off + d];
        }
    }

    // Backward: scatter-add output->grad rows into the accessed weight rows.
    if (weight->requires_grad()) {
        auto w = weight;
        auto node = make_node(output, {w});
        node->backward_fn = [w, token_ids, batch, seq_len, embed_dim=embed_dim](const TensorPtr& output) {
            for (int b = 0; b < batch; b++) {
                for (int t = 0; t < seq_len; t++) {
                    int id      = token_ids[b * seq_len + t];
                    int out_off = (b * seq_len + t) * embed_dim;
                    int w_off   = id * embed_dim;
                    for (int d = 0; d < embed_dim; d++)
                        w->grad()[w_off + d] += output->grad()[out_off + d];
                }
            }
        };
    }

    return output;
}

vector<TensorPtr> Embedding::parameters() {
    return {weight};
}
