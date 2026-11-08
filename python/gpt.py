"""
GPT (decoder-only transformer) built in Python from prometheus layers.

This is the same architecture as the C++ `prometheus.GPT`, but assembled the way
a user of the library would assemble it: by composing Embedding, TransformerBlock
and LayerNorm and wiring them with autograd ops. Nothing here reaches past the
public API into raw buffers.

    token_ids [S]
      -> tok_emb + pos_emb            [S, E]     (or RoPE, in which case pos_emb is unused)
      -> N x TransformerBlock         [S, E]     (causal, pre-norm, GELU FFN)
      -> LayerNorm                    [S, E]
      -> logits = x @ tok_emb.weight^T  [S, V]   (weight tying)

Weight tying uses `matmul_bt`, which multiplies by the transpose without
building it. It is differentiable, so gradients from the
output projection flow back into the same tensor the embedding lookup reads, so
`tok_emb.weight` appears exactly once in `parameters()`.

Usage:
    from prometheus.gpt import GPT
    model = GPT(vocab_size=256, max_seq_len=64, embed_dim=128, num_heads=4, num_layers=4)
    logits = model.forward([1, 2, 3])
    loss   = p.cross_entropy_sparse_seq(logits, [2, 3, 4])
    loss.backward()
"""

import math

import prometheus as p

from .models.base import Model


class GPT(Model):
    """Decoder-only transformer, GPT-2 style.

    Subclasses models.Model so it inherits visualise(), record_loss() and the
    train()/eval() cascade along with the forward()/parameters() contract.
    """

    def __init__(self, vocab_size, max_seq_len, embed_dim,
                 num_heads, num_layers, ff_dim=0, rope=False):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.rope = rope

        ffn_dim = ff_dim if ff_dim > 0 else 4 * embed_dim

        self.tok_emb = self.add("tok_emb", p.Embedding(vocab_size, embed_dim))
        self.pos_emb = self.add("pos_emb", p.Embedding(max_seq_len, embed_dim))
        self.blocks = [
            self.add("block%d" % (i + 1),
                     p.TransformerBlock(embed_dim, num_heads, ffn_dim,
                                        causal=True, rope=rope))
            for i in range(num_layers)
        ]
        self.ln_f = self.add("ln_f", p.LayerNorm(embed_dim))

        # Scale down weight init on the residual paths (GPT-2 paper, section 2.3):
        # W_o and ff2 divided by sqrt(2 * num_layers) so the residual stream
        # variance stays around 1 at initialisation.
        residual_scale = 1.0 / math.sqrt(2.0 * num_layers)
        for blk in self.blocks:
            _scale(blk.attn.W_o, residual_scale)
            _scale(blk.ff2.weights, residual_scale)

    def trace(self, token_ids):
        """Per-block activations, for the dashboard."""
        S = len(token_ids)
        E = self.embed_dim
        out = []

        ids = p.Tensor([1, S], [float(t) for t in token_ids])
        x = p.reshape(self.tok_emb.forward(ids), [S, E])
        out.append(("tok_emb", x))

        if not self.rope:
            pos = p.Tensor([1, S], [float(i) for i in range(S)])
            x = p.add(x, p.reshape(self.pos_emb.forward(pos), [S, E]))
            out.append(("pos_emb", x))

        for i, blk in enumerate(self.blocks):
            x = blk.forward(x)
            out.append(("block%d" % (i + 1), x))

        x = self.ln_f.forward(x)
        out.append(("ln_f", x))

        logits = p.matmul_bt(x, self.tok_emb.weight)
        out.append(("logits", logits))
        return out

    def forward(self, token_ids):
        """token_ids: sequence of ints, length S. Returns [S, vocab_size] logits."""
        S = len(token_ids)
        if S == 0:
            raise ValueError("token_ids must not be empty")
        if S > self.max_seq_len:
            raise ValueError(
                "sequence length {} exceeds max_seq_len {}".format(S, self.max_seq_len))
        E = self.embed_dim

        for t in token_ids:
            if not 0 <= t < self.vocab_size:
                raise ValueError("token id {} out of range".format(t))

        # Embedding expects [batch, seq] and returns [batch, seq, embed];
        # reshape drops the batch axis while keeping the gradient path.
        # reshape would be free, but a view carries no data of its own
        # and the blocks below read data directly, so it cannot go here yet.
        ids = p.Tensor([1, S], [float(t) for t in token_ids])
        x = p.reshape(self.tok_emb.forward(ids), [S, E])

        if not self.rope:
            pos = p.Tensor([1, S], [float(i) for i in range(S)])
            x = p.add(x, p.reshape(self.pos_emb.forward(pos), [S, E]))

        for blk in self.blocks:
            x = blk.forward(x)

        x = self.ln_f.forward(x)

        # Weight tying: logits = x @ tok_emb.weight^T, transpose never built
        return p.matmul_bt(x, self.tok_emb.weight)

    def parameters(self):
        """All learnable tensors.

        Overrides Model.parameters() rather than using the inherited walk:
        tok_emb.weight is tied to the output projection so it must appear
        exactly once, and pos_emb is unused when rope=True so including it
        would hand the optimizer a tensor that never receives a gradient.
        """
        params = [self.tok_emb.weight]
        if not self.rope:
            params.append(self.pos_emb.weight)
        for blk in self.blocks:
            params.extend(blk.parameters())
        params.extend(self.ln_f.parameters())
        return params


def _scale(tensor, factor):
    """Multiply a tensor's data in place."""
    data = tensor.data
    tensor.data = [v * factor for v in data]
