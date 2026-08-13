"""
GPT training on Shakespeare using tiktoken + prometheus.

Architecture (GPT-2 style):
  token_ids → Embedding + learned pos_emb → N × (causal MHA + GELU FFN, pre-norm)
            → LayerNorm → logits = x @ tok_emb.weight^T   [weight tying]

Loss: cross_entropy_sparse_seq on raw logits — no softmax step needed.

Tokenizer: cl100k_base with dense vocab remap for efficient small-corpus training.

Usage:
  python examples/gpt_shakespeare.py                     # trains on built-in snippet
  python examples/gpt_shakespeare.py path/to/text.txt   # trains on your file
"""

import sys, os, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tiktoken
import prometheus as p

# ── Config ────────────────────────────────────────────────────────────────────
EMBED    = 64      # embedding dimension
HEADS    = 4       # attention heads (EMBED must divide evenly)
LAYERS   = 2       # transformer blocks
SEQ_LEN  = 32      # training context window
BATCH    = 4       # sequences per gradient step
LR       = 3e-3
WD       = 0.1
EPOCHS   = 200
MAX_NORM = 1.0
ROPE     = True    # use RoPE instead of learned positional embeddings

random.seed(42)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding("cl100k_base")

# ── Corpus ────────────────────────────────────────────────────────────────────
DEFAULT_TEXT = (
    "To be or not to be that is the question "
    "Whether tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune "
    "Or to take arms against a sea of troubles "
    "And by opposing end them to die to sleep "
    "No more and by a sleep to say we end "
    "The heartache and the thousand natural shocks "
    "That flesh is heir to tis a consummation "
    "Devoutly to be wished to die to sleep "
    "To sleep perchance to dream ay there is the rub "
    "For in that sleep of death what dreams may come "
    "When we have shuffled off this mortal coil "
    "Must give us pause there is the respect "
    "That makes calamity of so long life "
    "For who would bear the whips and scorns of time "
    "The wronged love the proud mans contumely "
    "The pangs of despised love the laws delay "
    "The insolence of office and the spurns "
    "That patient merit of the unworthy takes "
    "When he himself might his quietus make "
    "With a bare bodkin who would fardels bear "
    "To grunt and sweat under a weary life "
    "But that the dread of something after death "
    "The undiscovered country from whose bourn "
    "No traveller returns puzzles the will "
    "And makes us rather bear those ills we have "
    "Than fly to others that we know not of "
    "Thus conscience does make cowards of us all "
    "And thus the native hue of resolution "
    "Is sicklied over with the pale cast of thought "
    "And enterprises of great pitch and moment "
    "With this regard their currents turn awry "
    "And lose the name of action "
) * 4  # repeat to get more training data

if len(sys.argv) > 1:
    with open(sys.argv[1], encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {sys.argv[1]}: {len(text):,} chars")
else:
    text = DEFAULT_TEXT


# ── TokenDataset — streaming chunker with dense vocab remap ───────────────────
class TokenDataset:
    """
    Tokenizes text with tiktoken, remaps to a dense local vocabulary (only tokens
    that actually appear in the corpus), and chunks into fixed-length windows.

    Dense remap: cl100k_base has 100k tokens but a short corpus uses only a few
    hundred. Keeping only the observed tokens shrinks the embedding and output
    head from [100k, E] to [local_vocab, E], which saves memory and speeds up
    training. The full cl100k_base tokenizer is still used for encode/decode.
    """

    def __init__(self, text, enc, seq_len=128, dense_remap=True):
        print(f"Tokenizing {len(text):,} chars...")
        raw_ids = enc.encode(text)

        if dense_remap:
            unique          = sorted(set(raw_ids))
            self.local2gpt  = unique                          # local → cl100k
            self.gpt2local  = {g: l for l, g in enumerate(unique)}
            self.token_ids  = [self.gpt2local[i] for i in raw_ids]
            self.vocab_size = len(unique)
        else:
            self.local2gpt  = list(range(enc.n_vocab))
            self.gpt2local  = {i: i for i in range(enc.n_vocab)}
            self.token_ids  = raw_ids
            self.vocab_size = enc.n_vocab

        self.seq_len = seq_len

        # Build non-overlapping windows of length seq_len + 1
        # (input = window[:-1], target = window[1:])
        self.windows = [
            self.token_ids[i: i + seq_len + 1]
            for i in range(0, len(self.token_ids) - seq_len, seq_len)
        ]
        print(f"vocab: {self.vocab_size} unique tokens "
              f"(cl100k has {enc.n_vocab:,})  |  windows: {len(self.windows)}")

    def shuffle(self):
        random.shuffle(self.windows)

    def __len__(self):
        return len(self.windows)

    def decode(self, local_ids):
        return enc.decode([self.local2gpt[i] for i in local_ids])


# ── Build dataset ─────────────────────────────────────────────────────────────
dataset  = TokenDataset(text, enc, seq_len=SEQ_LEN)
VOCAB    = dataset.vocab_size
MAXLEN   = SEQ_LEN + 1   # pos embedding table size

# ── Model ─────────────────────────────────────────────────────────────────────
model    = p.GPT(VOCAB, MAXLEN, EMBED, HEADS, LAYERS, rope=ROPE)
params   = model.parameters()
opt      = p.AdamW(params, lr=LR, weight_decay=WD)
sched    = p.CosineAnnealingLR(base_lr=LR, T_max=EPOCHS, min_lr=LR / 10)

n_params = sum(t.num_el() for t in params)
print(f"\nModel: {n_params:,} parameters  "
      f"(vocab={VOCAB}, embed={EMBED}, heads={HEADS}, layers={LAYERS}, seq={SEQ_LEN})\n")

# ── Training ──────────────────────────────────────────────────────────────────
print(f"{'epoch':>6}  {'loss':>8}  {'ppl':>8}  {'lr':>10}")
print("-" * 40)

for epoch in range(1, EPOCHS + 1):
    dataset.shuffle()
    total_loss = 0.0
    n_steps    = 0

    # Process windows in mini-batches (accumulate gradients across sequences)
    for batch_start in range(0, len(dataset), BATCH):
        batch = dataset.windows[batch_start: batch_start + BATCH]
        if not batch:
            continue

        batch_loss = None
        for window in batch:
            inp_ids = window[:-1]   # [seq_len] → feed to model
            tgt_ids = window[1:]    # [seq_len] → targets

            ids_tensor = p.Tensor([SEQ_LEN], [float(x) for x in inp_ids])
            logits     = model.forward(ids_tensor)       # [seq_len, vocab]
            loss       = p.cross_entropy_sparse_seq(logits, tgt_ids)

            batch_loss = loss if batch_loss is None else p.add(batch_loss, loss)

        # Average over batch
        if len(batch) > 1:
            batch_loss = p.divide(batch_loss, float(len(batch)))

        batch_loss.backward()
        p.clip_grad_norm(params, MAX_NORM)
        opt.step()
        opt.zero_grad()

        total_loss += batch_loss.data[0]
        n_steps    += 1

    opt.lr = sched.step()

    if epoch % 20 == 0 or epoch == 1:
        avg_loss = total_loss / max(n_steps, 1)
        ppl      = math.exp(min(avg_loss, 20))   # cap to avoid overflow
        print(f"{epoch:>6}  {avg_loss:>8.4f}  {ppl:>8.1f}  {opt.lr:>10.2e}")

# ── Generation ────────────────────────────────────────────────────────────────
def top_k_sample(logit_data, k=5, temperature=0.8):
    top = sorted(enumerate(logit_data), key=lambda x: x[1], reverse=True)[:k]
    ids, vals = zip(*top)
    exp_vals = [math.exp((v - max(vals)) / temperature) for v in vals]
    total    = sum(exp_vals)
    r, cum   = random.random(), 0.0
    for idx, prob in zip(ids, (e / total for e in exp_vals)):
        cum += prob
        if r <= cum:
            return idx
    return ids[-1]

print("\n--- generated text (KV cache, top-k=5, t=0.8) ---")
SEED     = "To be or not"
seed_ids = [dataset.gpt2local[i] for i in enc.encode(SEED) if i in dataset.gpt2local]
if not seed_ids:
    seed_ids = [random.randint(0, VOCAB - 1)]

generated = list(seed_ids)
print(f"seed: '{SEED}'")

# Prime the KV cache with all prompt tokens except the last.
# Each call consumes O(n) instead of O(n²) — linear scaling at inference.
cache = model.make_kv_cache()
for tid in seed_ids[:-1]:
    model.forward_cached(int(tid), cache)

# Generate autoregressively; each step only processes one new token.
# Stop if KV cache would overflow (or configure MAXLEN > SEQ_LEN at the top).
ctx_id = seed_ids[-1]
for _ in range(120):
    if cache.past_len >= MAXLEN - 1:
        break
    logits  = model.forward_cached(int(ctx_id), cache)   # [vocab_size]
    ctx_id  = top_k_sample(logits.data)
    generated.append(ctx_id)

print(dataset.decode(generated))
