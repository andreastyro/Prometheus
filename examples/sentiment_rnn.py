"""
Sentiment classification with a GRU.

A tiny vocabulary of 20 "words" (represented as integer IDs) is split:
    IDs 0–9:   "positive" words  (good, great, love, …)
    IDs 10–19: "negative" words  (bad, awful, hate, …)

Positive sentences are random sequences drawn from IDs 0–9;
negative sentences from IDs 10–19.
The GRU reads the whole sequence and classifies it as positive (1) or negative (0).

Demonstrates:
    - One-hot encoding for sequence input
    - GRU with forward_with_state to get the final hidden state
    - Binary classification on variable-length sequences
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import prometheus as p

random.seed(11)

VOCAB_SIZE  = 20
HIDDEN_SIZE = 32
N_PER_CLASS = 100
SEQ_LEN_MIN = 4
SEQ_LEN_MAX = 10

# ── Dataset ───────────────────────────────────────────────────────────────────
# Each sample is a list of token IDs and a label.
def make_sentence(cls):
    pool = list(range(0, 10) if cls == 1 else range(10, 20))
    length = random.randint(SEQ_LEN_MIN, SEQ_LEN_MAX)
    return [random.choice(pool) for _ in range(length)]

samples = []
for cls in range(2):
    for _ in range(N_PER_CLASS):
        samples.append((make_sentence(cls), float(cls)))

random.shuffle(samples)
n_train = int(len(samples) * 0.8)
train_set = samples[:n_train]
val_set   = samples[n_train:]

print(f"dataset: {n_train} train  {len(val_set)} val  |  vocab={VOCAB_SIZE}  2 classes")

# ── Helpers ───────────────────────────────────────────────────────────────────
def seq_to_tensor(token_ids):
    """One-hot encode a token sequence -> Tensor [seq_len, 1, vocab_size]."""
    seq_len = len(token_ids)
    flat    = [0.0] * (seq_len * VOCAB_SIZE)
    for t, tid in enumerate(token_ids):
        flat[t * VOCAB_SIZE + tid] = 1.0
    return p.Tensor([seq_len, 1, VOCAB_SIZE], flat)

def zero_state():
    return p.Tensor([1, HIDDEN_SIZE], [0.0] * HIDDEN_SIZE)

# ── Model ─────────────────────────────────────────────────────────────────────
gru    = p.GRU(VOCAB_SIZE, HIDDEN_SIZE)
linear = p.Linear(HIDDEN_SIZE, 1)

params    = gru.parameters() + linear.parameters()
optimizer = p.Adam(params, lr=0.005)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n{'epoch':>6}  {'loss':>8}  {'val_acc':>8}")
print("-" * 30)

for epoch in range(1, 41):
    total_loss = 0.0
    random.shuffle(train_set)

    for tokens, label in train_set:
        x    = seq_to_tensor(tokens)
        h0   = zero_state()
        out, h_n = gru.forward_with_state(x, h0)   # h_n: [1, hidden_size]

        logit = linear.forward(h_n)                 # [1, 1]
        pred  = p.sigmoid(logit)
        pred  = p.clip(pred, 1e-7, 1 - 1e-7)

        target = p.Tensor([1, 1], [label])
        loss   = p.bce_loss(pred, target)
        loss.backward()
        total_loss += loss.data[0]

        optimizer.step()
        optimizer.zero_grad()

    if epoch % 8 == 0:
        correct = 0
        for tokens, label in val_set:
            x        = seq_to_tensor(tokens)
            h0       = zero_state()
            _, h_n   = gru.forward_with_state(x, h0)
            logit    = linear.forward(h_n)
            pred_val = p.sigmoid(logit).data[0]
            if round(pred_val) == int(label):
                correct += 1
        val_acc = correct / len(val_set)
        avg_loss = total_loss / len(train_set)
        print(f"{epoch:>6}  {avg_loss:>8.4f}  {val_acc:>8.3f}")

# ── Final accuracy ────────────────────────────────────────────────────────────
correct = 0
for tokens, label in val_set:
    x      = seq_to_tensor(tokens)
    h0     = zero_state()
    _, h_n = gru.forward_with_state(x, h0)
    pred   = p.sigmoid(linear.forward(h_n)).data[0]
    if round(pred) == int(label):
        correct += 1
print(f"\nfinal val accuracy: {correct / len(val_set):.3f}")
