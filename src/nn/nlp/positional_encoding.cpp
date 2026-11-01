#include "ml/nn/nlp/positional_encoding.hpp"
#include "ml/tensor.hpp"
#include <cmath>

using namespace std;

PositionalEncoding::PositionalEncoding(int max_len, int embed_dim)
    : max_len(max_len), embed_dim(embed_dim) {}

TensorPtr PositionalEncoding::forward(TensorPtr input) {
    int seq_len = input->shape[0];
    auto output = make_shared<Tensor>(input->shape);
    output->data = input->data;

    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < embed_dim; i += 2) {
            // frequency gets exponentially smaller as i increases
            float freq = 1.0f / std::pow(10000.0f, (float)i / embed_dim);

            // even dim → sine (fast to slow)
            output->data[pos * embed_dim + i] += std::sin(pos * freq);

            // odd dim → cosine (same frequency, different phase)
            if (i + 1 < embed_dim)
                output->data[pos * embed_dim + i + 1] += std::cos(pos * freq);
        }
    }

    // y = x + const, so dy/dx = 1: pass the gradient straight through.
    // Without this node the graph is severed here and anything upstream
    // (typically the embedding table) would silently receive no gradient.
    if (input->requires_grad) {
        auto node = make_node(output, {input});

        node->backward_fn = [input, output]() {
            for (int i = 0; i < output->num_el(); i++)
                input->grad[i] += output->grad[i];
        };
    }

    return output;
}
