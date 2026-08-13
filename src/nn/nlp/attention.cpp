#include "ml/nn/nlp/attention.hpp"
#include "ml/autograd.hpp"
#include "ml/matmul_backend.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace std;

// RoPE: rotate each consecutive pair of dims by angle pos * θ_i,  θ_i = 10000^(-2i/D).
// v is [S, D], modified in-place. pos_offset shifts all position indices (for KV cache).
static void apply_rope(vector<float>& v, int S, int D, int pos_offset) {
    for (int t = 0; t < S; t++) {
        for (int i = 0; i < D / 2; i++) {
            float theta = powf(10000.0f, -2.0f * i / D);
            float angle = (t + pos_offset) * theta;
            float c = cosf(angle), s = sinf(angle);
            float v0 = v[t*D + 2*i], v1 = v[t*D + 2*i + 1];
            v[t*D + 2*i]     = v0*c - v1*s;
            v[t*D + 2*i + 1] = v0*s + v1*c;
        }
    }
}

// Backward through RoPE: applies the transpose (= inverse) rotation R(-θ) to the gradient.
static void apply_rope_backward(vector<float>& grad, int S, int D, int pos_offset) {
    for (int t = 0; t < S; t++) {
        for (int i = 0; i < D / 2; i++) {
            float theta = powf(10000.0f, -2.0f * i / D);
            float angle = (t + pos_offset) * theta;
            float c = cosf(angle), s = sinf(angle);
            float g0 = grad[t*D + 2*i], g1 = grad[t*D + 2*i + 1];
            grad[t*D + 2*i]     =  g0*c + g1*s;
            grad[t*D + 2*i + 1] = -g0*s + g1*c;
        }
    }
}

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads, bool causal, bool rope)
    : embed_dim(embed_dim), num_heads(num_heads), head_dim(embed_dim / num_heads),
      causal_(causal), rope_(rope) {
    if (embed_dim % num_heads != 0)
        throw runtime_error("MultiHeadAttention: embed_dim must be divisible by num_heads");

    W_q = Tensor::randn({embed_dim, embed_dim}); W_q->requires_grad = true;
    W_k = Tensor::randn({embed_dim, embed_dim}); W_k->requires_grad = true;
    W_v = Tensor::randn({embed_dim, embed_dim}); W_v->requires_grad = true;
    W_o = Tensor::randn({embed_dim, embed_dim}); W_o->requires_grad = true;
}

// Fused multi-head attention forward + backward.
//
// The old implementation sliced Q/K/V heads via raw data writes, which severed
// the autograd graph — W_q/W_k/W_v never received gradients.  This rewrite
// computes the entire forward analytically, saves all intermediates needed for
// the backward, and attaches a single GradNode that implements the full
// attention gradient in one lambda.
TensorPtr MultiHeadAttention::forward(TensorPtr input) {
    int S = input->shape[0];
    int E = embed_dim;
    int H = num_heads;
    int D = head_dim;
    float attn_scale = 1.0f / sqrtf((float)D);

    // ── Forward ──────────────────────────────────────────────────────────────
    // Q = input @ W_q,  K = input @ W_k,  V = input @ W_v   each [S, E]
    vector<float> Q_all(S * E), K_all(S * E), V_all(S * E);
    matmul_forward(S, E, E, input->data.data(), W_q->data.data(), Q_all.data());
    matmul_forward(S, E, E, input->data.data(), W_k->data.data(), K_all.data());
    matmul_forward(S, E, E, input->data.data(), W_v->data.data(), V_all.data());

    // Per-head slices and softmax weights saved for backward
    vector<vector<float>> Q_heads(H, vector<float>(S * D));
    vector<vector<float>> K_heads(H, vector<float>(S * D));
    vector<vector<float>> V_heads(H, vector<float>(S * D));
    vector<vector<float>> A_heads(H, vector<float>(S * S));

    vector<float> concat_all(S * E, 0.0f);

    for (int h = 0; h < H; h++) {
        int start = h * D;

        // Slice Q_h, K_h, V_h  [S, D]
        for (int t = 0; t < S; t++)
            for (int d = 0; d < D; d++) {
                Q_heads[h][t*D + d] = Q_all[t*E + start + d];
                K_heads[h][t*D + d] = K_all[t*E + start + d];
                V_heads[h][t*D + d] = V_all[t*E + start + d];
            }

        if (rope_) {
            apply_rope(Q_heads[h], S, D, 0);
            apply_rope(K_heads[h], S, D, 0);
        }

        // scores = Q_h @ K_h^T  [S, S] — transpose K_h into [D, S] for matmul
        vector<float> K_h_T(D * S);
        for (int t = 0; t < S; t++)
            for (int d = 0; d < D; d++)
                K_h_T[d*S + t] = K_heads[h][t*D + d];

        vector<float> scores(S * S, 0.0f);
        matmul_forward(S, S, D, Q_heads[h].data(), K_h_T.data(), scores.data());
        for (float& v : scores) v *= attn_scale;

        // Causal mask: future positions get ~0 weight after softmax
        if (causal_)
            for (int i = 0; i < S; i++)
                for (int j = i + 1; j < S; j++)
                    scores[i*S + j] = -1e9f;

        // Row-wise softmax → A_h [S, S]
        for (int i = 0; i < S; i++) {
            float max_v = *max_element(scores.data() + i*S, scores.data() + (i+1)*S);
            float sum = 0.0f;
            for (int j = 0; j < S; j++) {
                A_heads[h][i*S + j] = expf(scores[i*S + j] - max_v);
                sum += A_heads[h][i*S + j];
            }
            for (int j = 0; j < S; j++) A_heads[h][i*S + j] /= sum;
        }

        // O_h = A_h @ V_h  [S, D]
        vector<float> O_h(S * D, 0.0f);
        matmul_forward(S, D, S, A_heads[h].data(), V_heads[h].data(), O_h.data());

        for (int t = 0; t < S; t++)
            for (int d = 0; d < D; d++)
                concat_all[t*E + start + d] = O_h[t*D + d];
    }

    // output = concat @ W_o  [S, E]
    auto result = make_shared<Tensor>(vector<int>{S, E});
    matmul_forward(S, E, E, concat_all.data(), W_o->data.data(), result->data.data());

    // ── Backward ─────────────────────────────────────────────────────────────
    bool need_grad = input->requires_grad || W_q->requires_grad ||
                     W_k->requires_grad   || W_v->requires_grad || W_o->requires_grad;

    if (need_grad) {
        auto node = make_node(result, {input, W_q, W_k, W_v, W_o});

        bool rope = rope_;

        // Capture all intermediates and weight TensorPtrs by value.
        // TensorPtrs keep the weight tensors alive; vectors own their data.
        node->backward_fn = [
            input, W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, result,
            concat_all, Q_heads, K_heads, V_heads, A_heads,
            S, E, H, D, attn_scale, rope
        ]() {
            const float* grad_out = result->grad.data();

            // ── Output projection backward ────────────────────────────────
            // output = concat @ W_o:
            //   grad_concat = grad_out @ W_o^T
            //   grad_W_o   += concat^T @ grad_out
            vector<float> grad_concat(S * E, 0.0f);
            matmul_backward_a(S, E, E, grad_out, W_o->data.data(), grad_concat.data());
            if (W_o->requires_grad)
                matmul_backward_b(S, E, E, grad_out, concat_all.data(), W_o->grad.data());

            // Full gradient accumulators for the Q, K, V projections  [S, E]
            vector<float> grad_Q(S * E, 0.0f);
            vector<float> grad_K(S * E, 0.0f);
            vector<float> grad_V(S * E, 0.0f);

            // ── Per-head backward ─────────────────────────────────────────
            for (int h = 0; h < H; h++) {
                int start = h * D;
                const float* A_h = A_heads[h].data();
                const float* Q_h = Q_heads[h].data();
                const float* K_h = K_heads[h].data();
                const float* V_h = V_heads[h].data();

                // Gather grad_O_h from grad_concat  [S, D]
                vector<float> grad_O_h(S * D);
                for (int t = 0; t < S; t++)
                    for (int d = 0; d < D; d++)
                        grad_O_h[t*D + d] = grad_concat[t*E + start + d];

                // O_h = A_h @ V_h:
                //   grad_A_h [S,S] += grad_O_h [S,D] @ V_h^T [D,S]
                //   grad_V_h [S,D] += A_h^T [S,S] @ grad_O_h [S,D]
                vector<float> grad_A_h(S * S, 0.0f);
                matmul_backward_a(S, S, D, grad_O_h.data(), V_h, grad_A_h.data());

                vector<float> grad_V_h(S * D, 0.0f);
                matmul_backward_b(S, S, D, grad_O_h.data(), A_h, grad_V_h.data());

                // Softmax backward: grad_scores_h = A_h * (grad_A_h - rowsum(grad_A_h * A_h))
                // Causal-masked positions have A_h[i,j] ≈ 0, so their gradients vanish naturally.
                vector<float> grad_scores(S * S);
                for (int i = 0; i < S; i++) {
                    float rowsum = 0.0f;
                    for (int j = 0; j < S; j++)
                        rowsum += grad_A_h[i*S + j] * A_h[i*S + j];
                    for (int j = 0; j < S; j++)
                        grad_scores[i*S + j] = A_h[i*S + j] * (grad_A_h[i*S + j] - rowsum);
                }

                // Scale backward (scores were multiplied by attn_scale before softmax)
                for (float& g : grad_scores) g *= attn_scale;

                // scores = Q_h @ K_h^T:
                //   grad_Q_h [S,D] = grad_scores [S,S] @ K_h [S,D]
                //   grad_K_h [S,D] = grad_scores^T [S,S] @ Q_h [S,D]
                vector<float> grad_Q_h(S * D, 0.0f);
                matmul_forward(S, D, S, grad_scores.data(), K_h, grad_Q_h.data());

                vector<float> grad_scores_T(S * S);
                for (int i = 0; i < S; i++)
                    for (int j = 0; j < S; j++)
                        grad_scores_T[j*S + i] = grad_scores[i*S + j];
                vector<float> grad_K_h(S * D, 0.0f);
                matmul_forward(S, D, S, grad_scores_T.data(), Q_h, grad_K_h.data());

                // RoPE backward: Q_h and K_h in the saved heads are already rotated.
                // grad_Q_h/grad_K_h are in the rotated space — apply inverse rotation
                // to bring them back into the original (pre-rotation) Q/K space.
                if (rope) {
                    apply_rope_backward(grad_Q_h, S, D, 0);
                    apply_rope_backward(grad_K_h, S, D, 0);
                }

                // Scatter head gradients into full [S, E] accumulators
                for (int t = 0; t < S; t++)
                    for (int d = 0; d < D; d++) {
                        grad_Q[t*E + start + d] += grad_Q_h[t*D + d];
                        grad_K[t*E + start + d] += grad_K_h[t*D + d];
                        grad_V[t*E + start + d] += grad_V_h[t*D + d];
                    }
            }

            // ── Input projection backward ─────────────────────────────────
            // Q = input @ W_q:
            //   grad_W_q   += input^T @ grad_Q
            //   grad_input += grad_Q @ W_q^T   (and similarly for K, V)
            if (W_q->requires_grad)
                matmul_backward_b(S, E, E, grad_Q.data(), input->data.data(), W_q->grad.data());
            if (W_k->requires_grad)
                matmul_backward_b(S, E, E, grad_K.data(), input->data.data(), W_k->grad.data());
            if (W_v->requires_grad)
                matmul_backward_b(S, E, E, grad_V.data(), input->data.data(), W_v->grad.data());

            if (input->requires_grad) {
                matmul_backward_a(S, E, E, grad_Q.data(), W_q->data.data(), input->grad.data());
                matmul_backward_a(S, E, E, grad_K.data(), W_k->data.data(), input->grad.data());
                matmul_backward_a(S, E, E, grad_V.data(), W_v->data.data(), input->grad.data());
            }
        };
    }

    return result;
}

vector<TensorPtr> MultiHeadAttention::parameters() {
    return {W_q, W_k, W_v, W_o};
}
