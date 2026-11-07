#include "ml/nn/nlp/attention.hpp"
#include "ml/autograd.hpp"
#include "ml/matmul_backend.hpp"
#include <cmath>
#include <cstring>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <string>   // to_string; MSVC does not get it transitively

using namespace std;

// RoPE: rotate each consecutive pair of dims by angle pos * θ_i,  θ_i = 10000^(-2i/D).
// v is [S, D], modified in-place. pos_offset shifts all position indices (for KV cache).
static void apply_rope_at(float* v, int S, int D, int pos_offset) {
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
static void apply_rope_backward_at(float* grad, int S, int D, int pos_offset) {
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

static inline void apply_rope(vector<float>& v, int S, int D, int off) {
    apply_rope_at(v.data(), S, D, off);
}
static inline void apply_rope_backward(vector<float>& g, int S, int D, int off) {
    apply_rope_backward_at(g.data(), S, D, off);
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
    return forward(input, 1);
}

TensorPtr MultiHeadAttention::forward(TensorPtr input, int batch) {
    int rows = input->shape[0];
    if (batch <= 0 || rows % batch != 0)
        throw runtime_error("MultiHeadAttention: row count " + to_string(rows) +
                            " is not divisible by batch " + to_string(batch));
    int B = batch;
    int S = rows / B;
    int E = embed_dim;
    int H = num_heads;
    int D = head_dim;
    float attn_scale = 1.0f / sqrtf((float)D);

    // ── Forward ──────────────────────────────────────────────────────────────
    // Q = input @ W_q,  K = input @ W_k,  V = input @ W_v   each [S, E]
    // Three [rows, E] x [E, E] gemms are small enough that BLAS cannot
    // amortise its packing over them -- measured at 72 GFLOPS against 188 for
    // the same work as one [rows, E] x [E, 3E] gemm, a 2.6x difference. So
    // pack the three weights side by side and issue a single call. The pack is
    // 3*E*E floats and is dwarfed by what it saves.
    const int E3 = 3 * E;
    vector<float> W_qkv((size_t)E * E3);
    for (int e = 0; e < E; e++) {
        memcpy(&W_qkv[(size_t)e*E3        ], &W_q->data[(size_t)e*E], E*sizeof(float));
        memcpy(&W_qkv[(size_t)e*E3 + E    ], &W_k->data[(size_t)e*E], E*sizeof(float));
        memcpy(&W_qkv[(size_t)e*E3 + 2*E  ], &W_v->data[(size_t)e*E], E*sizeof(float));
    }
    vector<float> QKV((size_t)rows * E3);
    matmul_forward(rows, E3, E, input->data.data(), W_qkv.data(), QKV.data());

    // Per-head slices and softmax weights saved for backward
    vector<vector<float>> Q_heads(H, vector<float>(rows * D));
    vector<vector<float>> K_heads(H, vector<float>(rows * D));
    vector<vector<float>> V_heads(H, vector<float>(rows * D));
    // one [S, S] attention matrix per head per sequence
    vector<vector<float>> A_heads(H, vector<float>(B * S * S));

    vector<float> concat_all(rows * E, 0.0f);

    // Slicing and rotation are per head, and each head owns its own buffer,
    // so this splits across threads with no sharing.
    #pragma omp parallel for schedule(static) if(H > 1)
    for (int h = 0; h < H; h++) {
        const int start = h * D;

        // Slice Q_h, K_h, V_h for every row  [rows, D]
        for (int t = 0; t < rows; t++)
            for (int d = 0; d < D; d++) {
                Q_heads[h][t*D + d] = QKV[t*E3 + start + d];
                K_heads[h][t*D + d] = QKV[t*E3 + E + start + d];
                V_heads[h][t*D + d] = QKV[t*E3 + 2*E + start + d];
            }

        // RoPE positions restart at every sequence, so rotate each slice
        // separately rather than letting position run across the batch.
        if (rope_)
            for (int b = 0; b < B; b++) {
                apply_rope_at(Q_heads[h].data() + b*S*D, S, D, 0);
                apply_rope_at(K_heads[h].data() + b*S*D, S, D, 0);
            }
    }

    // Scores, softmax and the value product are independent for every
    // (head, sequence) pair -- 48 of them at 6 heads and batch 8, every bit
    // of it scalar work that was running on a single core while the matmuls
    // around it used all 16. Flattened into one loop because MSVC's OpenMP
    // 2.0 has no collapse(2).
    const int HB = H * B;
    #pragma omp parallel for schedule(static) if(HB > 1)
    for (int hb = 0; hb < HB; hb++) {
        const int h = hb / B, b = hb % B;
        const int start = h * D;
        {
            const float* Q_b = Q_heads[h].data() + b*S*D;
            const float* K_b = K_heads[h].data() + b*S*D;
            const float* V_b = V_heads[h].data() + b*S*D;
            float* A_b = A_heads[h].data() + b*S*S;

            // scores = Q_b @ K_b^T  [S, S], with the transpose handed to the
            // kernel as a flag. Building K_b^T explicitly ran once per head per
            // sequence per layer - 288 copies and 9.4 MB per forward pass at
            // batch 8, for a matrix that is only read.
            vector<float> scores(S * S, 0.0f);
            matmul_forward_bt_serial(S, S, D, Q_b, K_b, scores.data());
            for (float& v : scores) v *= attn_scale;

            // Causal mask: future positions get ~0 weight after softmax
            if (causal_)
                for (int i = 0; i < S; i++)
                    for (int j = i + 1; j < S; j++)
                        scores[i*S + j] = -1e9f;

            // Row-wise softmax → A_b [S, S]
            for (int i = 0; i < S; i++) {
                float max_v = *max_element(scores.data() + i*S, scores.data() + (i+1)*S);
                float sum = 0.0f;
                for (int j = 0; j < S; j++) {
                    A_b[i*S + j] = expf(scores[i*S + j] - max_v);
                    sum += A_b[i*S + j];
                }
                for (int j = 0; j < S; j++) A_b[i*S + j] /= sum;
            }

            // O_b = A_b @ V_b  [S, D]
            vector<float> O_b(S * D, 0.0f);
            matmul_forward_serial(S, D, S, A_b, V_b, O_b.data());

            for (int t = 0; t < S; t++)
                for (int d = 0; d < D; d++)
                    concat_all[(b*S + t)*E + start + d] = O_b[t*D + d];
        }
    }

    // output = concat @ W_o  [S, E]
    auto result = make_shared<Tensor>(vector<int>{rows, E});
    matmul_forward(rows, E, E, concat_all.data(), W_o->data.data(), result->data.data());

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
            // moved, not copied: at [1024, 384] with 6 heads these buffers
            // are about 9 MB together, and copying them per layer per forward
            // cost more than some of the matmuls they exist to support
            concat_all  = std::move(concat_all),
            Q_heads     = std::move(Q_heads),
            K_heads     = std::move(K_heads),
            V_heads     = std::move(V_heads),
            A_heads     = std::move(A_heads),
            W_qkv       = std::move(W_qkv),
            rows, B, S, E, E3, H, D, attn_scale, rope
        ]() {
            const float* grad_out = result->grad.data();

            // ── Output projection backward ────────────────────────────────
            // output = concat @ W_o:
            //   grad_concat = grad_out @ W_o^T
            //   grad_W_o   += concat^T @ grad_out
            vector<float> grad_concat(rows * E, 0.0f);
            matmul_backward_a(rows, E, E, grad_out, W_o->data.data(), grad_concat.data());
            if (W_o->requires_grad)
                matmul_backward_b(rows, E, E, grad_out, concat_all.data(), W_o->grad.data());

            // Full gradient accumulators for the Q, K, V projections  [rows, E]
            // one interleaved [rows, 3E] accumulator, laid out to match
            // W_qkv so the projection backward is also a single pair of gemms
            vector<float> grad_QKV((size_t)rows * E3, 0.0f);

            // ── Per-head backward ─────────────────────────────────────────
            // Same independence as the forward: every (head, sequence) pair
            // writes only its own slice of grad_QKV.
            const int HB = H * B;
            #pragma omp parallel for schedule(static) if(HB > 1)
            for (int hb = 0; hb < HB; hb++) {
                const int h = hb / B, b = hb % B;
                const int start = h * D;
              {
                const float* A_h = A_heads[h].data() + b*S*S;
                const float* Q_h = Q_heads[h].data() + b*S*D;
                const float* K_h = K_heads[h].data() + b*S*D;
                const float* V_h = V_heads[h].data() + b*S*D;

                // Gather grad_O_h from grad_concat  [S, D]
                vector<float> grad_O_h(S * D);
                for (int t = 0; t < S; t++)
                    for (int d = 0; d < D; d++)
                        grad_O_h[t*D + d] = grad_concat[(b*S + t)*E + start + d];

                // O_h = A_h @ V_h:
                //   grad_A_h [S,S] += grad_O_h [S,D] @ V_h^T [D,S]
                //   grad_V_h [S,D] += A_h^T [S,S] @ grad_O_h [S,D]
                vector<float> grad_A_h(S * S, 0.0f);
                matmul_backward_a_serial(S, S, D, grad_O_h.data(), V_h, grad_A_h.data());

                vector<float> grad_V_h(S * D, 0.0f);
                matmul_backward_b_serial(S, S, D, grad_O_h.data(), A_h, grad_V_h.data());

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
                matmul_forward_serial(S, D, S, grad_scores.data(), K_h, grad_Q_h.data());

                // grad_scores^T @ Q_h is A^T @ B, which is the shape
                // matmul_backward_b already computes, so the [S,S] transpose
                // buffer goes away too - 18.9 MB per backward pass at batch 8.
                vector<float> grad_K_h(S * D, 0.0f);
                matmul_backward_b_serial(S, S, D, Q_h, grad_scores.data(), grad_K_h.data());

                // RoPE backward: Q_h and K_h in the saved heads are already rotated.
                // grad_Q_h/grad_K_h are in the rotated space — apply inverse rotation
                // to bring them back into the original (pre-rotation) Q/K space.
                if (rope) {
                    apply_rope_backward_at(grad_Q_h.data(), S, D, 0);
                    apply_rope_backward_at(grad_K_h.data(), S, D, 0);
                }

                // Scatter head gradients into the full [rows, E] accumulators
                for (int t = 0; t < S; t++)
                    for (int d = 0; d < D; d++) {
                        const size_t base = (size_t)(b*S + t)*E3 + start + d;
                        grad_QKV[base          ] += grad_Q_h[t*D + d];
                        grad_QKV[base + E      ] += grad_K_h[t*D + d];
                        grad_QKV[base + 2*E    ] += grad_V_h[t*D + d];
                    }
              }
            }

            // ── Input projection backward ─────────────────────────────────
            // Q = input @ W_q:
            //   grad_W_q   += input^T @ grad_Q
            //   grad_input += grad_Q @ W_q^T   (and similarly for K, V)
            if (W_q->requires_grad || W_k->requires_grad || W_v->requires_grad) {
                // one gemm for all three weight gradients, then split the
                // [E, 3E] result back into the three parameter tensors
                vector<float> W_qkv_grad((size_t)E * E3, 0.0f);
                matmul_backward_b(rows, E, E3, grad_QKV.data(),
                                  input->data.data(), W_qkv_grad.data());
                for (int e = 0; e < E; e++)
                    for (int j = 0; j < E; j++) {
                        const size_t src = (size_t)e*E3 + j;
                        const size_t dst = (size_t)e*E  + j;
                        if (W_q->requires_grad) W_q->grad[dst] += W_qkv_grad[src];
                        if (W_k->requires_grad) W_k->grad[dst] += W_qkv_grad[src + E];
                        if (W_v->requires_grad) W_v->grad[dst] += W_qkv_grad[src + 2*E];
                    }
            }

            if (input->requires_grad)
                matmul_backward_a(rows, E, E3, grad_QKV.data(), W_qkv.data(),
                                  input->grad.data());
        };
    }

    return result;
}

vector<TensorPtr> MultiHeadAttention::parameters() {
    return {W_q, W_k, W_v, W_o};
}
