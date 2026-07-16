#include "ml/nn/embedding.hpp"
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

    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq_len; t++) {
            int id = static_cast<int>(input->data[b * seq_len + t]);

            if (id < 0 || id >= vocab_size)
                throw runtime_error("Embedding: token id " + to_string(id) + " out of range [0, " + to_string(vocab_size) + ")");

            // copy row `id` from weight into output[b][t]
            int out_offset    = (b * seq_len + t) * embed_dim;
            int weight_offset = id * embed_dim;
            for (int d = 0; d < embed_dim; d++)
                output->data[out_offset + d] = weight->data[weight_offset + d];
        }
    }

    return output;
}

vector<TensorPtr> Embedding::parameters() {
    return {weight};
}
