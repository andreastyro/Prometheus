#include "ml/nn/nlp/attention.hpp"
#include "ml/ops.hpp"
#include "ml/autograd.hpp"
#include "ml/matmul_backend.hpp"
#include <cmath>
#include "ml/fastmath.hpp"
#include <cstring>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <string>   // to_string; MSVC does not get it transitively

using namespace std;

// RoPE: rotate each consecutive pair of dims by angle pos * θ_i,  θ_i = 10000^(-2i/D).
// v is [S, D], modified in-place. pos_offset shifts all position indices (for KV cache).
static void apply_rope_at(float* v, int S, int D, int pos_offset, int ld) {
    for (int t = 0; t < S; t++) {
        for (int i = 0; i < D / 2; i++) {
            float theta = powf(10000.0f, -2.0f * i / D);
            float angle = (t + pos_offset) * theta;
            float c = cosf(angle), s = sinf(angle);
            float v0 = v[t*ld + 2*i], v1 = v[t*ld + 2*i + 1];
            v[t*ld + 2*i]     = v0*c - v1*s;
            v[t*ld + 2*i + 1] = v0*s + v1*c;
        }
    }
}

// Backward through RoPE: applies the transpose (= inverse) rotation R(-θ) to the gradient.
static void apply_rope_backward_at(float* grad, int S, int D, int pos_offset, int ld) {
    for (int t = 0; t < S; t++) {
        for (int i = 0; i < D / 2; i++) {
            float theta = powf(10000.0f, -2.0f * i / D);
            float angle = (t + pos_offset) * theta;
            float c = cosf(angle), s = sinf(angle);
            float g0 = grad[t*ld + 2*i], g1 = grad[t*ld + 2*i + 1];
            grad[t*ld + 2*i]     =  g0*c + g1*s;
            grad[t*ld + 2*i + 1] = -g0*s + g1*c;
        }
    }
}

static inline void apply_rope(vector<float>& v, int S, int D, int off) {
    apply_rope_at(v.data(), S, D, off, D);
}
static inline void apply_rope_backward(vector<float>& g, int S, int D, int off) {
    apply_rope_backward_at(g.data(), S, D, off, D);
}

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads, bool causal, bool rope)
    : embed_dim(embed_dim), num_heads(num_heads), head_dim(embed_dim / num_heads),
      causal_(causal), rope_(rope) {
    if (embed_dim % num_heads != 0)
        throw runtime_error("MultiHeadAttention: embed_dim must be divisible by num_heads");

    W_q = Tensor::randn({embed_dim, embed_dim}); W_q->set_requires_grad(true);
    W_k = Tensor::randn({embed_dim, embed_dim}); W_k->set_requires_grad(true);
    W_v = Tensor::randn({embed_dim, embed_dim}); W_v->set_requires_grad(true);
    W_o = Tensor::randn({embed_dim, embed_dim}); W_o->set_requires_grad(true);
}

// Fused multi-head attention forward + backward.
//
// The old implementation sliced Q/K/V heads via raw data writes, which severed
// the autograd graph — W_q/W_k/W_v never received gradients.  This rewrite
// computes the entire forward analytically, saves all intermediates needed for
// the backward, and attaches a single GradNode that implements the full
// attention gradient in one lambda.
TensorPtr MultiHeadAttention::forward(TensorPtr input) {
    if (input->dim() == 3) {
        const int64_t B = input->size(0), S = input->size(1);
        auto rows = reshape(checked_input("MultiHeadAttention", input), {B * S, input->size(2)});
        return reshape(forward_rows(rows, (int)B), {B, S, (int64_t)embed_dim});
    }
    if (input->dim() == 2)
        return forward_rows(checked_input("MultiHeadAttention", input), 1);
    throw runtime_error("MultiHeadAttention: input must be [batch, seq, embed] or [seq, embed]");
}

TensorPtr MultiHeadAttention::forward_rows(TensorPtr input, int batch) {
    int rows = input->shape()[0];
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
        memcpy(&W_qkv[(size_t)e*E3        ], &W_q->data()[(size_t)e*E], E*sizeof(float));
        memcpy(&W_qkv[(size_t)e*E3 + E    ], &W_k->data()[(size_t)e*E], E*sizeof(float));
        memcpy(&W_qkv[(size_t)e*E3 + 2*E  ], &W_v->data()[(size_t)e*E], E*sizeof(float));
    }
    vector<float> QKV((size_t)rows * E3);
    matmul_forward(rows, E3, E, input->data(), W_qkv.data(), QKV.data());

    // Heads are read straight out of the fused QKV buffer: head h of row t
    // starts at QKV[t*E3 + h*D], so every gemm below takes a row stride of E3
    // rather than a dense [S, D] copy. This used to copy Q, K and V into
    // per-head buffers and every head's output back into concat_all; both
    // copies, and the three per-head buffers, are gone.
    //
    // softmax weights, one [S, S] matrix per head per sequence, for backward
    vector<vector<float>> A_heads(H, vector<float>((size_t)B * S * S));
    vector<float> concat_all((size_t)rows * E, 0.0f);

    // RoPE rotates Q and K in place, per sequence so positions restart at 0
    if (rope_) {
        #pragma omp parallel for schedule(static) if(H > 1)
        for (int h = 0; h < H; h++)
            for (int b = 0; b < B; b++) {
                float* q = QKV.data() + (size_t)b*S*E3 + h*D;
                apply_rope_at(q,     S, D, 0, E3);
                apply_rope_at(q + E, S, D, 0, E3);
            }
    }

    {
    // BLAS pinned to one thread for the per-head gemms only; the projections
    // either side are large and want every core
    SerialBlasScope serial_blas;
    const int HB = H * B;
    #pragma omp parallel for schedule(static) if(HB > 1)
    for (int hb = 0; hb < HB; hb++) {
        const int h = hb / B, b = hb % B;
        const float* Q_b = QKV.data() + (size_t)b*S*E3 + h*D;
        const float* K_b = Q_b + E;
        const float* V_b = Q_b + 2*E;
        float* A_b = A_heads[h].data() + (size_t)b*S*S;

        // scores = Q_b @ K_b^T  [S, S]
        vector<float> scores((size_t)S * S);
        gemm_serial(false, true, S, S, D, Q_b, E3, K_b, E3, 0.0f, scores.data(), S);
        for (float& v : scores) v *= attn_scale;

        // Causal mask: future positions get ~0 weight after softmax
        if (causal_)
            for (int i = 0; i < S; i++)
                for (int j = i + 1; j < S; j++)
                    scores[i*S + j] = -1e9f;

        // exp, sum and scale as three separate loops: a float sum in the
        // same loop as the exp is a reduction the compiler will not reorder,
        // and it would hold the exp back to one lane too
        for (int i = 0; i < S; i++) {
            const float max_v = *max_element(scores.data() + i*S, scores.data() + (i+1)*S);
            float* row = A_b + i*S;
            const float* sc = scores.data() + i*S;
            for (int j = 0; j < S; j++) row[j] = fastmath::exp(sc[j] - max_v);
            float sum = 0.0f;
            for (int j = 0; j < S; j++) sum += row[j];
            const float inv = 1.0f / sum;
            for (int j = 0; j < S; j++) row[j] *= inv;
        }

        // O_b = A_b @ V_b, written straight into this head's columns
        gemm_serial(false, false, S, D, S, A_b, S, V_b, E3, 0.0f,
                    concat_all.data() + (size_t)b*S*E + h*D, E);
    }
    }   // serial BLAS scope ends

    // output = concat @ W_o  [S, E]
    auto result = make_shared<Tensor>(Shape{rows, E});
    matmul_forward(rows, E, E, concat_all.data(), W_o->data(), result->data());

    // ── Backward ─────────────────────────────────────────────────────────────
    bool need_grad = input->requires_grad() || W_q->requires_grad() ||
                     W_k->requires_grad()   || W_v->requires_grad() || W_o->requires_grad();

    if (need_grad) {
        auto node = make_node(result, {input, W_q, W_k, W_v, W_o});
        node->name = "MultiHeadAttention::forward_rows";
        autograd::attach_reference(*node, [B, S, E, H, D, attn_scale, use_rope = rope_, causal = causal_](const std::vector<TensorPtr>& in) -> TensorPtr {
            const int64_t b = B, s = S, e = E, h = H, d = D;
            auto x3 = reshape(in[0], {b, s, e});
            auto heads = [&](const TensorPtr& W) {                  // [B, H, S, D]
                return permute(reshape(matmul(x3, W), {b, s, h, d}), {0, 2, 1, 3});
            };
            auto Q = heads(in[1]), K = heads(in[2]), V = heads(in[3]);
            if (use_rope) {
                // RoPE as a constant per-position rotation: v' = v @ R_t
                vector<float> R((size_t)S * D * D, 0.0f);
                for (int t = 0; t < S; t++)
                    for (int i = 0; i < D / 2; i++) {
                        const float theta = powf(10000.0f, -2.0f * i / D), angle = t * theta;
                        const float c = cosf(angle), sn = sinf(angle);
                        float* r = R.data() + (size_t)t * D * D;
                        r[(2 * i) * D + 2 * i] = c;      r[(2 * i) * D + 2 * i + 1] = sn;
                        r[(2 * i + 1) * D + 2 * i] = -sn; r[(2 * i + 1) * D + 2 * i + 1] = c;
                    }
                auto Rt = make_shared<Tensor>(Shape{s, d, d}, R);
                Q = reshape(matmul(reshape(Q, {b, h, s, 1, d}), Rt), {b, h, s, d});
                K = reshape(matmul(reshape(K, {b, h, s, 1, d}), Rt), {b, h, s, d});
            }
            auto scores = multiply(attn_scale, matmul(Q, transpose(K)));    // [B, H, S, S]
            if (causal) {
                vector<float> m((size_t)S * S, 0.0f);
                for (int i = 0; i < S; i++) for (int j = i + 1; j < S; j++) m[(size_t)i * S + j] = -1e9f;
                scores = add(scores, make_shared<Tensor>(Shape{s, s}, m));
            }
            auto O = matmul(softmax(scores), V);                            // [B, H, S, D]
            O = reshape(permute(O, {0, 2, 1, 3}), {b * s, e});
            return matmul(O, in[4]);
        });

        bool rope = rope_;

        // Capture all intermediates and weight TensorPtrs by value.
        // TensorPtrs keep the weight tensors alive; vectors own their data.
        node->backward_fn = [
            input, W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o,
            // moved, not copied: at [1024, 384] with 6 heads these buffers
            // are about 9 MB together, and copying them per layer per forward
            // cost more than some of the matmuls they exist to support
            concat_all  = std::move(concat_all),
            QKV         = std::move(QKV),
            A_heads     = std::move(A_heads),
            W_qkv       = std::move(W_qkv),
            rows, B, S, E, E3, H, D, attn_scale, rope
        ](const TensorPtr& result) {
            const float* grad_out = result->grad();

            // ── Output projection backward ────────────────────────────────
            // output = concat @ W_o:
            //   grad_concat = grad_out @ W_o^T
            //   grad_W_o   += concat^T @ grad_out
            vector<float> grad_concat(rows * E, 0.0f);
            matmul_backward_a(rows, E, E, grad_out, W_o->data(), grad_concat.data());
            if (W_o->requires_grad())
                matmul_backward_b(rows, E, E, grad_out, concat_all.data(), W_o->grad());

            // Full gradient accumulators for the Q, K, V projections  [rows, E]
            // one interleaved [rows, 3E] accumulator, laid out to match
            // W_qkv so the projection backward is also a single pair of gemms
            vector<float> grad_QKV((size_t)rows * E3, 0.0f);

            // ── Per-head backward ─────────────────────────────────────────
            // Same layout as the forward: heads are read from the saved QKV
            // and from grad_concat with row strides, and each (head, sequence)
            // pair writes its gradients straight into its own block of
            // grad_QKV, which no other iteration touches.
            {
            SerialBlasScope serial_blas;
            const int HB = H * B;
            #pragma omp parallel for schedule(static) if(HB > 1)
            for (int hb = 0; hb < HB; hb++) {
                const int h = hb / B, b = hb % B;
                const size_t qkv_off = (size_t)b*S*E3 + h*D;
                const float* A_h = A_heads[h].data() + (size_t)b*S*S;
                const float* Q_h = QKV.data() + qkv_off;
                const float* K_h = Q_h + E;
                const float* V_h = Q_h + 2*E;
                const float* gO  = grad_concat.data() + (size_t)b*S*E + h*D;
                float* gQ = grad_QKV.data() + qkv_off;
                float* gK = gQ + E;
                float* gV = gQ + 2*E;

                // O = A @ V:  grad_A = grad_O @ V^T,  grad_V = A^T @ grad_O
                vector<float> grad_A((size_t)S * S);
                gemm_serial(false, true, S, S, D, gO, E, V_h, E3, 0.0f, grad_A.data(), S);
                gemm_serial(true, false, S, D, S, A_h, S, gO, E, 1.0f, gV, E3);

                // Softmax backward, with the attn_scale applied before the
                // softmax folded in: grad_scores = scale * A * (grad_A - rowsum)
                // Causal-masked positions have A ~ 0, so their gradients vanish.
                vector<float> grad_scores((size_t)S * S);
                for (int i = 0; i < S; i++) {
                    float rowsum = 0.0f;
                    for (int j = 0; j < S; j++)
                        rowsum += grad_A[i*S + j] * A_h[i*S + j];
                    for (int j = 0; j < S; j++)
                        grad_scores[i*S + j] = attn_scale * A_h[i*S + j] * (grad_A[i*S + j] - rowsum);
                }

                // scores = Q @ K^T:  grad_Q = grad_scores @ K,  grad_K = grad_scores^T @ Q
                gemm_serial(false, false, S, D, S, grad_scores.data(), S, K_h, E3, 1.0f, gQ, E3);
                gemm_serial(true,  false, S, D, S, grad_scores.data(), S, Q_h, E3, 1.0f, gK, E3);

                // The saved Q and K are rotated, so their gradients are in the
                // rotated space: apply the inverse rotation to bring them back.
                if (rope) {
                    apply_rope_backward_at(gQ, S, D, 0, E3);
                    apply_rope_backward_at(gK, S, D, 0, E3);
                }
            }
            }   // serial BLAS scope ends

            // ── Input projection backward ─────────────────────────────────
            // Q = input @ W_q:
            //   grad_W_q   += input^T @ grad_Q
            //   grad_input += grad_Q @ W_q^T   (and similarly for K, V)
            if (W_q->requires_grad() || W_k->requires_grad() || W_v->requires_grad()) {
                // one gemm for all three weight gradients, then split the
                // [E, 3E] result back into the three parameter tensors
                vector<float> W_qkv_grad((size_t)E * E3, 0.0f);
                matmul_backward_b(rows, E, E3, grad_QKV.data(),
                                  input->data(), W_qkv_grad.data());
                for (int e = 0; e < E; e++)
                    for (int j = 0; j < E; j++) {
                        const size_t src = (size_t)e*E3 + j;
                        const size_t dst = (size_t)e*E  + j;
                        if (W_q->requires_grad()) W_q->grad()[dst] += W_qkv_grad[src];
                        if (W_k->requires_grad()) W_k->grad()[dst] += W_qkv_grad[src + E];
                        if (W_v->requires_grad()) W_v->grad()[dst] += W_qkv_grad[src + 2*E];
                    }
            }

            if (input->requires_grad())
                matmul_backward_a(rows, E, E3, grad_QKV.data(), W_qkv.data(),
                                  input->grad());
        };
    }

    return result;
}

vector<TensorPtr> MultiHeadAttention::parameters() {
    return {W_q, W_k, W_v, W_o};
}
