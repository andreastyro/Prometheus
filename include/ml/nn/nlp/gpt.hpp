#pragma once
#include "ml/nn/module.hpp"
#include "ml/nn/nlp/embedding.hpp"
#include "ml/nn/nlp/layernorm.hpp"
#include "ml/nn/nlp/transformer.hpp"
#include <vector>

// KVCache holds the key and value vectors accumulated during autoregressive inference.
// K[l] and V[l] are flat float vectors of length (past_len * embed_dim) for layer l.
// Append a new token by calling GPT::forward_cached — it extends the cache in place.
// Inference-only: no gradient tracking, no autograd overhead.
struct KVCache {
    std::vector<std::vector<float>> K; // K[l] = concatenated K vectors for layer l
    std::vector<std::vector<float>> V;
    int past_len = 0;

    explicit KVCache(int num_layers) : K(num_layers), V(num_layers) {}
    void reset() {
        for (auto& k : K) k.clear();
        for (auto& v : V) v.clear();
        past_len = 0;
    }
};

// GPT — decoder-only transformer (GPT-2 architecture).
//
// Layout:
//   token_ids [seq_len]
//       → tok_emb [seq_len, embed_dim]  +  pos_emb [seq_len, embed_dim]  (or RoPE)
//       → x [seq_len, embed_dim]
//       → N × TransformerBlock (causal, pre-norm, GELU FFN)
//       → LayerNorm (final)
//       → logits = x @ tok_emb.weight^T   [seq_len, vocab_size]  (weight tying)
//
// Weight tying: the output projection reuses tok_emb.weight, so gradients from
// the output head accumulate into the same tensor as the embedding lookup.
//
// RoPE mode (rope=true): positional information is encoded by rotating Q and K
// inside each attention head. pos_emb is still allocated but not used — set
// rope=true to discard it from the parameter count too.
//
// forward() input:  flat TensorPtr of shape [seq_len] with float-encoded token IDs
// forward() output: [seq_len, vocab_size] raw logits (feed to cross_entropy_sparse_seq)
//
// forward_cached() processes one token at a time using a KVCache for O(n) inference.
class GPT {
public:
    int vocab_size_;
    int max_seq_len_;
    int embed_dim_;
    int num_heads_;
    int num_layers_;
    bool rope_;

    Embedding tok_emb;
    Embedding pos_emb; // learned positional embeddings; unused when rope_=true
    std::vector<TransformerBlock> blocks;
    LayerNorm ln_f;

    // ff_dim defaults to 4 * embed_dim, matching GPT-2 exactly.
    // rope=true uses Rotary Position Embedding instead of learned absolute positions.
    GPT(int vocab_size, int max_seq_len, int embed_dim,
        int num_heads, int num_layers, int ff_dim = 0, bool rope = false);

    // Full-sequence training forward. Returns [seq_len, vocab_size] logits.
    TensorPtr forward(TensorPtr token_ids);

    // Incremental inference forward — processes one token, updates cache.
    // Returns [vocab_size] logits for the new position.
    // Call GPT::make_kv_cache() to initialise a fresh cache before generation.
    TensorPtr forward_cached(int token_id, KVCache& cache);

    // Returns an empty KVCache pre-sized for this model's number of layers.
    KVCache make_kv_cache() const { return KVCache(num_layers_); }

    // All parameters — tok_emb.weight appears once (shared with lm_head).
    std::vector<TensorPtr> parameters();
};
