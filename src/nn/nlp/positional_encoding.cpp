#include "ml/nn/nlp/positional_encoding.hpp"
#include "ml/ops.hpp"
#include "ml/tensor.hpp"
#include <cmath>

using namespace std;

PositionalEncoding::PositionalEncoding(int max_len, int embed_dim)
    : max_len(max_len), embed_dim(embed_dim) {}

TensorPtr PositionalEncoding::forward(TensorPtr input) {
    int seq_len = input->shape()[0];
    auto output = make_shared<Tensor>(input->shape());
    output->set_data(input->to_vector());

    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < embed_dim; i += 2) {
            // frequency gets exponentially smaller as i increases
            float freq = 1.0f / std::pow(10000.0f, (float)i / embed_dim);

            // even dim → sine (fast to slow)
            output->data()[pos * embed_dim + i] += std::sin(pos * freq);

            // odd dim → cosine (same frequency, different phase)
            if (i + 1 < embed_dim)
                output->data()[pos * embed_dim + i + 1] += std::cos(pos * freq);
        }
    }

    // y = x + const, so dy/dx = 1: pass the gradient straight through.
    // Without this node the graph is severed here and anything upstream
    // (typically the embedding table) would silently receive no gradient.
    if (input->requires_grad()) {
        auto node = make_node(output, {input});
        node->name = "PositionalEncoding::forward";
        autograd::attach_reference(*node, [pe = make_shared<Tensor>(output->shape(), [&]{ auto o = output->to_vector(); auto x = input->to_vector(); for (size_t i = 0; i < o.size(); i++) o[i] -= x[i]; return o; }())](const std::vector<TensorPtr>& in) -> TensorPtr {
            // the encoding is a constant added to the input
            return add(in[0], pe);
        });

        node->backward_fn = [input](const TensorPtr& output) {
            for (int i = 0; i < output->num_el(); i++)
                input->grad()[i] += output->grad()[i];
        };
    }

    return output;
}
