"""
MNIST-style digit classifier using fully-connected layers.

Trains a 3-layer MLP on synthetic 784-dimensional data (same shape as
flattened 28x28 grayscale MNIST images) to classify digits 0-9.

Demonstrates:
    - Multi-class classification with cross-entropy loss
    - Dropout for regularisation
    - Accuracy metric on a held-out validation set
"""

import sys, os, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import prometheus as p

random.seed(42)

# ── Synthetic dataset ─────────────────────────────────────────────────────────
# Each class has a weak signal: feature[class_id] is biased upward.
# Everything else is Gaussian noise. Models must find the signal.
N_CLASSES  = 10
N_PER_CLS  = 60       # 60 samples per class = 600 total
N_FEATURES = 784      # 28×28 pixels flattened

X_all, y_all = [], []
for cls in range(N_CLASSES):
    # Each class has a unique prototype across all 784 features.
    # Feature f for class c = sin(f * (c+1) * 0.01) — distinct waveform per class.
    # Noise std=0.5 is small relative to the prototype amplitude (~1.0).
    prototype = [math.sin(f * (cls + 1) * 0.02) for f in range(N_FEATURES)]

    for _ in range(N_PER_CLS):
        features = [prototype[f] + random.gauss(0.0, 0.3) for f in range(N_FEATURES)]
        X_all.append(features)
        one_hot = [0.0] * N_CLASSES
        one_hot[cls] = 1.0
        y_all.append(one_hot)

n_total = len(X_all)
n_train = int(n_total * 0.8)

idx = list(range(n_total))
random.shuffle(idx)
train_idx = idx[:n_train]
val_idx   = idx[n_train:]

def make_tensor(rows):
    flat = [v for row in rows for v in row]
    return p.Tensor([len(rows), len(rows[0])], flat)

X_train = make_tensor([X_all[i] for i in train_idx])
y_train = make_tensor([y_all[i] for i in train_idx])
X_val   = make_tensor([X_all[i] for i in val_idx])
y_val   = make_tensor([y_all[i] for i in val_idx])

print(f"dataset: {n_train} train  {len(val_idx)} val  |  {N_FEATURES} features  {N_CLASSES} classes")

# ── Model ─────────────────────────────────────────────────────────────────────
layers = [
    p.Linear(784, 256, "xavier"),
    p.ReLU(),
    p.Dropout(0.2),
    p.Linear(256, 128, "xavier"),
    p.ReLU(),
    p.Linear(128, N_CLASSES, "xavier"),
    p.Softmax(),
]
model = p.Sequential(layers)

optimizer = p.Adam(model.parameters(), lr=0.001)
scheduler = p.StepLR(base_lr=0.001, step_size=20, gamma=0.5)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n{'epoch':>6}  {'loss':>8}  {'val_acc':>8}")
print("-" * 30)

for epoch in range(1, 61):
    out  = model.forward(X_train)
    out  = p.clip(out, 1e-7, 1.0)            # guard log(0) in cross-entropy
    loss = p.cross_entropy_loss(out, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    optimizer.lr = scheduler.step()

    if epoch % 10 == 0:
        # Dropout off for evaluation: recreate without dropout
        val_out = model.forward(X_val)
        preds   = p.argmax(val_out)
        labels  = p.argmax(y_val)
        correct = sum(1 for i in range(len(val_idx))
                      if int(preds.data[i]) == int(labels.data[i]))
        val_acc = correct / len(val_idx)
        print(f"{epoch:>6}  {loss.data[0]:>8.4f}  {val_acc:>8.3f}")

# ── Final metrics ─────────────────────────────────────────────────────────────
val_out = model.forward(X_val)
preds   = p.argmax(val_out)
labels  = p.argmax(y_val)
correct = sum(1 for i in range(len(val_idx))
              if int(preds.data[i]) == int(labels.data[i]))
print(f"\nfinal val accuracy: {correct / len(val_idx):.3f}")
