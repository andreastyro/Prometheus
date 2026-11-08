#include "ml/nn/nlp/embedding.hpp"
#include "ml/ops.hpp"
#include <utility>
#include <algorithm>
#include "ml/matmul_backend.hpp"
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

    const int rows = batch * seq_len;
    vector<int> token_ids(rows);
    for (int r = 0; r < rows; r++) {
        const int64_t id = ids[r];
        if (id < 0 || id >= vocab_size)
            throw runtime_error("Embedding: token id " + to_string(id)
                                + " out of range [0, " + to_string(vocab_size) + ")");
        token_ids[r] = (int)id;
    }

    // gather: every output row is independent
    {
        const float* W = weight->data();
        float* O = output->data();
        const int E = embed_dim;
        #pragma omp parallel for schedule(static) if((int64_t)rows * E >= 65536)
        for (int r = 0; r < rows; r++)
            std::copy(W + (int64_t)token_ids[r] * E, W + (int64_t)(token_ids[r] + 1) * E,
                      O + (int64_t)r * E);
    }

    // Backward: scatter-add output->grad rows into the rows that were looked
    // up. The same token can appear many times, so splitting by row would
    // race on its weight row; splitting the embedding dimension instead gives
    // each thread its own columns of every row.
    if (weight->requires_grad()) {
        auto w = weight;
        auto node = make_node(output, {w});
        node->name = "Embedding";
        autograd::attach_reference(*node, [tok = token_ids, E = embed_dim, oshape = output->shape()](const std::vector<TensorPtr>& in) -> TensorPtr {
            auto idx = make_shared<vector<int64_t>>(tok.size() * (size_t)E);
            for (size_t r = 0; r < tok.size(); r++)
                for (int d = 0; d < E; d++) (*idx)[r * E + d] = (int64_t)tok[r] * E + d;
            return gather_flat(in[0], idx, oshape);
        });
        node->backward_fn = [w, token_ids = std::move(token_ids), rows, E = embed_dim]
                            (const TensorPtr& output) {
            const float* G = output->grad();
            float* GW = w->grad();
            const int T = std::max(1, std::min(matmul_max_threads(), E / 16));
            #pragma omp parallel for schedule(static) if((int64_t)rows * E >= 65536)
            for (int q = 0; q < T; q++) {
                const int lo = E * q / T, hi = E * (q + 1) / T;
                for (int r = 0; r < rows; r++) {
                    float* dst = GW + (int64_t)token_ids[r] * E;
                    const float* src = G + (int64_t)r * E;
                    for (int d = lo; d < hi; d++) dst[d] += src[d];
                }
            }
        };
    }
    return output;
}

vector<TensorPtr> Embedding::parameters() {
    return {weight};
}
