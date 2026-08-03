"""
Binary classifier on two Gaussian clusters.

Two classes are placed at different centres in 2D space.
A small MLP learns the decision boundary between them.

Demonstrates:
    - Binary cross-entropy loss
    - Accuracy, precision, recall, F1
    - Sigmoid output for binary classification
"""

import sys, os, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import prometheus as p

random.seed(3)

# ── Two-cluster dataset ───────────────────────────────────────────────────────
# Class 0: centred at (-2, -2)
# Class 1: centred at ( 2,  2)
N_PER_CLASS = 150

X_all, y_all = [], []
for cls in range(2):
    cx = -2.0 + cls * 4.0
    cy = -2.0 + cls * 4.0
    for _ in range(N_PER_CLASS):
        x = random.gauss(cx, 1.2)
        y = random.gauss(cy, 1.2)
        X_all.append([x, y])
        y_all.append([float(cls)])

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

print(f"dataset: {n_train} train  {len(val_idx)} val  |  2 features  2 classes")

# ── Model ─────────────────────────────────────────────────────────────────────
layers = [
    p.Linear(2, 16),
    p.ReLU(),
    p.Linear(16, 8),
    p.ReLU(),
    p.Linear(8, 1),
    p.Sigmoid(),
]
model = p.Sequential(layers)
optimizer = p.Adam(model.parameters(), lr=0.01)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n{'epoch':>6}  {'loss':>8}  {'val_acc':>8}")
print("-" * 30)

for epoch in range(1, 101):
    out  = p.clip(model.forward(X_train), 1e-7, 1 - 1e-7)
    loss = p.bce_loss(out, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 20 == 0:
        val_out  = model.forward(X_val)
        val_pred = p.Tensor(val_out.shape, [round(v) for v in val_out.data])
        val_acc  = p.accuracy(val_pred, y_val)
        print(f"{epoch:>6}  {loss.data[0]:>8.4f}  {val_acc:>8.3f}")

# ── Final metrics ─────────────────────────────────────────────────────────────
val_out  = model.forward(X_val)
val_pred = p.Tensor(val_out.shape, [round(v) for v in val_out.data])

acc  = p.accuracy( val_pred, y_val)
prec = p.precision(val_pred, y_val)
rec  = p.recall(   val_pred, y_val)
f1   = p.f1_score( val_pred, y_val)

print(f"\naccuracy:  {acc:.3f}")
print(f"precision: {prec:.3f}")
print(f"recall:    {rec:.3f}")
print(f"f1 score:  {f1:.3f}")
