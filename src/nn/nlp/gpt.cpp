#include "ml/nn/nlp/gpt.hpp"
#include "ml/autograd.hpp"
#include "ml/matmul_backend.hpp"
#include <cmath>
#include <string>
#include <stdexcept>
#include <algorithm>

using namespace std;

// ── Inference helpers (no autograd) ──────────────────────────────────────────

static void ln_arr(const float* x, float* y, const float* gamma, const float* beta,
                   int N, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < N; i++) mean += x[i];
    mean /= N;
    float var = 0.0f;
    for (int i = 0; i < N; i++) { float d = x[i] - mean; var += d*d; }
    var /= N;
    float is = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < N; i++)
        y[i] = gamma[i] * (x[i] - mean) * is + beta[i];
}

static void gelu_arr(const float* x, float* y, int n) {
    const float k = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float t = tanhf(k * (v + 0.044715f * v*v*v));
        y[i] = 0.5f * v * (1.0f + t);
    }
}

// y = x @ W + b,  W stored as [in, out] (Linear convention), single-row x [in]
static void linear_arr(const float* x, float* y, const float* W, const float* b,
                        int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; i++) sum += x[i] * W[i * out_dim + o];
        y[o] = sum;
    }
}

// Rotate a single D-length vector by RoPE at position `pos`
static void rope_rotate(float* v, int D, int pos) {
    for (int i = 0; i < D / 2; i++) {
        float theta = powf(10000.0f, -2.0f * i / D);
        float angle = pos * theta;
        float c = cosf(angle), s = sinf(angle);
        float v0 = v[2*i], v1 = v[2*i + 1];
        v[2*i]     = v0*c - v1*s;
        v[2*i + 1] = v0*s + v1*c;
    }
}

// ─────────────────────────────────────────────────────────────────────────────

GPT::GPT(int vocab_size, int max_seq_len, int embed_dim,
         int num_heads, int num_layers, int ff_dim, bool rope)
    : vocab_size_(vocab_size),
      max_seq_len_(max_seq_len),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      num_layers_(num_layers),
      rope_(rope),
      tok_emb(vocab_size, embed_dim),
      pos_emb(max_seq_len, embed_dim),
      ln_f(embed_dim) {

    if (embed_dim % num_heads != 0)
        throw runtime_error("GPT: embed_dim must be divisible by num_heads");
    if (max_seq_len <= 0)
        throw runtime_error("GPT: max_seq_len must be positive");

    int ffn_dim = (ff_dim > 0) ? ff_dim : 4 * embed_dim;
    blocks.reserve(num_layers);
    for (int i = 0; i < num_layers; i++)
        blocks.emplace_back(embed_dim, num_heads, ffn_dim, /*causal=*/true, rope);

    // Scale down weight init for residual paths (GPT-2 paper, §2.3):
    // W_o and ff2 weights divided by sqrt(2 * num_layers) so that at
    // initialisation the residual stream variance stays ≈1.
    float residual_scale = 1.0f / sqrtf(2.0f * (float)num_layers);
    for (auto& blk : blocks) {
        for (float& v : blk.attn.W_o->data) v *= residual_scale;
        for (float& v : blk.ff2.weights->data) v *= residual_scale;
    }
}

TensorPtr GPT::forward(TensorPtr token_ids) {
    int S = token_ids->num_el();
    int E = embed_dim_;
    int V = vocab_size_;

    if (S > max_seq_len_)
        throw runtime_error("GPT::forward: sequence length " + to_string(S)
                            + " exceeds max_seq_len " + to_string(max_seq_len_));

    // ── Input embedding (fused tok + pos, with gradient tracking) ────────────
    // Out[t, d] = tok_emb.weight[tok_id, d] + pos_emb.weight[t, d]
    auto x = make_shared<Tensor>(vector<int>{S, E});
    vector<int> tok_ids(S);
    for (int t = 0; t < S; t++) {
        tok_ids[t] = (int)token_ids->data[t];
        int tid     = tok_ids[t];
        if (tid < 0 || tid >= V)
            throw runtime_error("GPT: token id " + to_string(tid) + " out of range");
        for (int d = 0; d < E; d++)
            x->data[t*E + d] = tok_emb.weight->data[tid*E + d]
                              + pos_emb.weight->data[t  *E + d];
    }

    {
        auto tw = tok_emb.weight;
        auto pw = pos_emb.weight;
        auto node = make_node(x, {tw, pw});
        node->backward_fn = [tw, pw, x, tok_ids, S, E]() {
            for (int t = 0; t < S; t++) {
                int tid = tok_ids[t];
                for (int d = 0; d < E; d++) {
                    float g = x->grad[t*E + d];
                    tw->grad[tid*E + d] += g;
                    pw->grad[t  *E + d] += g;
                }
            }
        };
    }

    // ── Transformer blocks ────────────────────────────────────────────────────
    for (auto& blk : blocks)
        x = blk.forward(x);

    // ── Final LayerNorm ───────────────────────────────────────────────────────
    x = ln_f.forward(x);

    // ── Output projection (weight-tied to tok_emb) ────────────────────────────
    // logits[t, v] = Σ_d x[t, d] * tok_emb.weight[v, d]
    //              = x @ tok_emb.weight^T,   tok_emb.weight [V, E]
    // Implemented as: logits = x @ W_T  where W_T = tok_emb.weight^T [E, V]
    auto logits = make_shared<Tensor>(vector<int>{S, V});

    auto tw = tok_emb.weight;
    vector<float> W_T(E * V);
    for (int v = 0; v < V; v++)
        for (int d = 0; d < E; d++)
            W_T[d*V + v] = tw->data[v*E + d];

    matmul_forward(S, V, E, x->data.data(), W_T.data(), logits->data.data());

    // Register backward for the weight-tied projection
    {
        auto node = make_node(logits, {x, tw});
        node->backward_fn = [x, tw, logits, W_T, S, E, V]() {
            const float* gl = logits->grad.data();

            // grad_x += grad_logits @ W    (W_T [E,V] used as b in backward_a)
            if (x->requires_grad)
                matmul_backward_a(S, E, V, gl, W_T.data(), x->grad.data());

            // grad_W[v,d] += Σ_t grad_logits[t,v] * x[t,d]
            //   = (grad_logits^T @ x)^T
            // Computed via backward_b into W_T_grad [E,V] then transposed into tw->grad
            if (tw->requires_grad) {
                vector<float> W_T_grad(E * V, 0.0f);
                matmul_backward_b(S, E, V, gl, x->data.data(), W_T_grad.data());
                for (int v = 0; v < V; v++)
                    for (int d = 0; d < E; d++)
                        tw->grad[v*E + d] += W_T_grad[d*V + v];
            }
        };
    }

    return logits;
}

vector<TensorPtr> GPT::parameters() {
    vector<TensorPtr> params;
    params.push_back(tok_emb.weight);
    if (!rope_) params.push_back(pos_emb.weight);
    for (auto& blk : blocks)
        for (auto& p : blk.parameters())
            params.push_back(p);
    for (auto& p : ln_f.parameters())
        params.push_back(p);
    // lm_head weight is tok_emb.weight — already listed above (weight tying)
    return params;
}

// ── KV-cache inference ────────────────────────────────────────────────────────
//
// Processes one token at a time without autograd.  Each call:
//   1. Embeds token + position (or just token when using RoPE)
//   2. For each block: applies norm → computes new Q/K/V → appends K,V to cache
//      → attends new Q against full K_cache → applies W_o → residual → FFN
//   3. Final norm → weight-tied output projection
//   4. Returns [vocab_size] logits; cache.past_len is incremented.
//
// KV cache memory per layer: 2 × embed_dim × past_len × 4 bytes.
TensorPtr GPT::forward_cached(int token_id, KVCache& cache) {
    int E = embed_dim_;
    int V_sz = vocab_size_;
    int H = num_heads_;
    int D = embed_dim_ / H;
    int pos = cache.past_len;

    if (pos >= max_seq_len_)
        throw runtime_error("GPT::forward_cached: KV cache full (max_seq_len="
                            + to_string(max_seq_len_) + ")");
    if (token_id < 0 || token_id >= V_sz)
        throw runtime_error("GPT::forward_cached: token id " + to_string(token_id)
                            + " out of range");

    // ── Embedding ─────────────────────────────────────────────────────────────
    vector<float> x(E);
    const float* tw = tok_emb.weight->data.data();
    for (int d = 0; d < E; d++) x[d] = tw[token_id * E + d];

    if (!rope_) {
        const float* pw = pos_emb.weight->data.data();
        for (int d = 0; d < E; d++) x[d] += pw[pos * E + d];
    }

    // ── Transformer blocks ────────────────────────────────────────────────────
    vector<float> x_norm(E), Q(E), K_new(E), V_new(E), concat(E), attn_proj(E);

    for (int l = 0; l < num_layers_; l++) {
        auto& blk   = blocks[l];
        auto& Kcache = cache.K[l];
        auto& Vcache = cache.V[l];

        const float* Wq = blk.attn.W_q->data.data();
        const float* Wk = blk.attn.W_k->data.data();
        const float* Wv = blk.attn.W_v->data.data();
        const float* Wo = blk.attn.W_o->data.data();

        // Pre-norm 1
        ln_arr(x.data(), x_norm.data(),
               blk.norm1.gamma->data.data(), blk.norm1.beta->data.data(),
               E, blk.norm1.eps);

        // Q, K, V projections for this single token  (W is [E, E] = [in, out])
        linear_arr(x_norm.data(), Q.data(),     Wq, nullptr, E, E);
        linear_arr(x_norm.data(), K_new.data(), Wk, nullptr, E, E);
        linear_arr(x_norm.data(), V_new.data(), Wv, nullptr, E, E);

        // Apply RoPE per head to Q and K_new at position `pos`
        if (rope_) {
            for (int h = 0; h < H; h++) {
                rope_rotate(Q.data()     + h*D, D, pos);
                rope_rotate(K_new.data() + h*D, D, pos);
            }
        }

        // Append new K and V to the cache
        Kcache.insert(Kcache.end(), K_new.begin(), K_new.end());
        Vcache.insert(Vcache.end(), V_new.begin(), V_new.end());
        int cache_len = pos + 1; // total tokens in cache after appending

        // Per-head attention: Q [D] × K_cache [cache_len, D] → O_h [D]
        float attn_scale = 1.0f / sqrtf((float)D);
        fill(concat.begin(), concat.end(), 0.0f);

        for (int h = 0; h < H; h++) {
            // scores[t] = (Q_h · K_cache_h[t]) * attn_scale
            vector<float> scores(cache_len, 0.0f);
            for (int t = 0; t < cache_len; t++) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += Q[h*D + d] * Kcache[t*E + h*D + d];
                scores[t] = dot * attn_scale;
            }

            // Softmax
            float max_v = *max_element(scores.begin(), scores.end());
            float sum = 0.0f;
            for (float& sc : scores) { sc = expf(sc - max_v); sum += sc; }
            for (float& sc : scores) sc /= sum;

            // O_h [D] = Σ_t scores[t] * V_cache_h[t]
            for (int t = 0; t < cache_len; t++)
                for (int d = 0; d < D; d++)
                    concat[h*D + d] += scores[t] * Vcache[t*E + h*D + d];
        }

        // Output projection: attn_proj = concat @ W_o  (W_o is [E, E] = [in, out])
        linear_arr(concat.data(), attn_proj.data(), Wo, nullptr, E, E);

        // Residual: x = x + attn_proj
        for (int d = 0; d < E; d++) x[d] += attn_proj[d];

        // Pre-norm 2
        ln_arr(x.data(), x_norm.data(),
               blk.norm2.gamma->data.data(), blk.norm2.beta->data.data(),
               E, blk.norm2.eps);

        // FFN: ff1 [E → ff_dim], gelu, ff2 [ff_dim → E], residual
        int ff_dim = blk.ff1.weights->shape[1]; // ff1.weights is [E, ff_dim]
        vector<float> ff1_out(ff_dim), ff1_act(ff_dim), ff2_out(E);

        linear_arr(x_norm.data(), ff1_out.data(),
                   blk.ff1.weights->data.data(), blk.ff1.bias->data.data(), E, ff_dim);
        gelu_arr(ff1_out.data(), ff1_act.data(), ff_dim);
        linear_arr(ff1_act.data(), ff2_out.data(),
                   blk.ff2.weights->data.data(), blk.ff2.bias->data.data(), ff_dim, E);

        for (int d = 0; d < E; d++) x[d] += ff2_out[d];
    }

    // ── Final LayerNorm ───────────────────────────────────────────────────────
    ln_arr(x.data(), x.data(),
           ln_f.gamma->data.data(), ln_f.beta->data.data(), E, ln_f.eps);

    // ── Weight-tied output projection: logits[v] = Σ_d x[d] * tok_emb.weight[v, d] ─
    auto logits = make_shared<Tensor>(vector<int>{V_sz});
    const float* tw2 = tok_emb.weight->data.data();
    for (int v = 0; v < V_sz; v++) {
        float dot = 0.0f;
        for (int d = 0; d < E; d++) dot += x[d] * tw2[v*E + d];
        logits->data[v] = dot;
    }

    cache.past_len++;
    return logits;
}
