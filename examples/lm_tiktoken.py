"""
Token-level language model using cl100k_base (GPT-4 tokenizer).

Instead of one-hot encoding individual characters, we use tiktoken to convert
text into token IDs, then look them up in an Embedding table.  This is the
exact pipeline every modern LLM uses.

For training on a small corpus we remap only the tokens that appear in the
data to a dense local vocabulary — the same trick nanoGPT uses.  The full
cl100k_base tokenizer is still used for encoding / decoding, preserving
compatibility with the GPT-4 token space.

Architecture:
    text → cl100k_base → token_ids → remap → Embedding(local_vocab, EMBED)
         → LSTM(EMBED, HIDDEN) → Linear(HIDDEN, local_vocab) → next-token

Demonstrates:
    - cl100k_base tokenizer (GPT-4 tokenizer, 100,277 tokens)
    - Embedding as the input representation (not one-hot)
    - Dense-vocab remapping for efficient training on small corpora
    - AdamW + gradient clipping + cosine LR schedule
    - Autoregressive top-k generation
"""

import sys, os, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tiktoken
import prometheus as p

random.seed(42)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding("cl100k_base")

# ── Corpus ────────────────────────────────────────────────────────────────────
CORPUS_TEXT = (
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
)

raw_ids = enc.encode(CORPUS_TEXT)

# ── Dense vocab remap ──────────────────────────────────────────────────────────
# Only keep the tokens that actually appear. Maps cl100k IDs → local 0..N-1.
# This keeps the embedding table and output head small without changing the
# tokenizer — nanoGPT uses the same trick.
unique_ids   = sorted(set(raw_ids))
local_to_gpt = unique_ids                          # local_id → cl100k id
gpt_to_local = {g: l for l, g in enumerate(local_to_gpt)}
token_ids    = [gpt_to_local[i] for i in raw_ids]
VOCAB        = len(unique_ids)

print(f"corpus: {len(CORPUS_TEXT)} chars → {len(token_ids)} tokens")
print(f"cl100k vocab: {enc.n_vocab:,}  |  corpus vocab: {VOCAB} unique tokens")

def decode_local(local_ids):
    return enc.decode([local_to_gpt[i] for i in local_ids])

# ── Build chunks for truncated BPTT ───────────────────────────────────────────
CHUNK  = 16
chunks = []
for i in range(0, len(token_ids) - CHUNK, CHUNK):
    chunks.append((token_ids[i:i+CHUNK], token_ids[i+1:i+CHUNK+1]))

print(f"chunks: {len(chunks)} × {CHUNK} tokens")

# ── Helpers ───────────────────────────────────────────────────────────────────
def token_tensor(tid):
    """Single local token ID → [1, 1] tensor for Embedding lookup."""
    return p.Tensor([1, 1], [float(tid)])

def target_onehot(tid):
    """Single local token ID → one-hot [1, VOCAB]."""
    flat = [0.0] * VOCAB
    flat[tid] = 1.0
    return p.Tensor([1, VOCAB], flat)

def zeros_state(size):
    return p.Tensor([1, size], [0.0] * size)

def top_k_sample(logits_data, k=5, temperature=0.8):
    top = sorted(enumerate(logits_data), key=lambda x: x[1], reverse=True)[:k]
    ids, vals = zip(*top)
    scaled = [math.exp(v / temperature) for v in vals]
    total  = sum(scaled)
    scaled = [v / total for v in scaled]
    r, cum = random.random(), 0.0
    for i, prob in zip(ids, scaled):
        cum += prob
        if r <= cum:
            return i
    return ids[-1]

# ── Model ─────────────────────────────────────────────────────────────────────
EMBED  = 64
HIDDEN = 128

embedding = p.Embedding(VOCAB, EMBED)
lstm      = p.LSTM(EMBED, HIDDEN)
linear    = p.Linear(HIDDEN, VOCAB, "xavier")

params    = embedding.parameters() + lstm.parameters() + linear.parameters()
optimizer = p.AdamW(params, lr=3e-3, weight_decay=0.01)
scheduler = p.CosineAnnealingLR(base_lr=3e-3, T_max=300, min_lr=3e-4)

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS = 300
print(f"\n{'epoch':>6}  {'loss/tok':>10}")
print("-" * 22)

for epoch in range(1, EPOCHS + 1):
    random.shuffle(chunks)
    total_loss = 0.0
    n_tokens   = 0

    for inp_ids, tgt_ids in chunks:
        h = zeros_state(HIDDEN)
        c = zeros_state(HIDDEN)

        emb        = embedding.forward(token_tensor(inp_ids[0]))
        x          = emb.reshape([1, 1, EMBED])
        _, h, c    = lstm.forward_with_state(x, h, c)
        logits     = linear.forward(h)
        probs      = p.clip(p.softmax(logits), 1e-7, 1.0)
        chunk_loss = p.cross_entropy_loss(probs, target_onehot(tgt_ids[0]))

        for t in range(1, len(inp_ids)):
            emb        = embedding.forward(token_tensor(inp_ids[t]))
            x          = emb.reshape([1, 1, EMBED])
            _, h, c    = lstm.forward_with_state(x, h, c)
            logits     = linear.forward(h)
            probs      = p.clip(p.softmax(logits), 1e-7, 1.0)
            step_loss  = p.cross_entropy_loss(probs, target_onehot(tgt_ids[t]))
            chunk_loss = p.add(chunk_loss, step_loss)

        chunk_loss.backward()
        p.clip_grad_norm(params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += chunk_loss.data[0]
        n_tokens   += len(inp_ids)

    optimizer.lr = scheduler.step()

    if epoch % 60 == 0:
        print(f"{epoch:>6}  {total_loss / n_tokens:>10.4f}")

# ── Generation ────────────────────────────────────────────────────────────────
print("\n--- generated text ---")
SEED     = "To be or not"
seed_ids = [gpt_to_local[i] for i in enc.encode(SEED) if i in gpt_to_local]
print(f"seed: '{SEED}'")

h = zeros_state(HIDDEN)
c = zeros_state(HIDDEN)
for tid in seed_ids:
    emb     = embedding.forward(token_tensor(tid))
    x       = emb.reshape([1, 1, EMBED])
    _, h, c = lstm.forward_with_state(x, h, c)

generated = list(seed_ids)
for _ in range(80):
    emb     = embedding.forward(token_tensor(generated[-1]))
    x       = emb.reshape([1, 1, EMBED])
    _, h, c = lstm.forward_with_state(x, h, c)
    logits  = linear.forward(h)
    probs   = p.softmax(logits)
    next_id = top_k_sample(probs.data, k=5, temperature=0.8)
    generated.append(next_id)

print(decode_local(generated))
