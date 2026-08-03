"""
House price regression — tabular data.

Predicts a synthetic house price from 5 features using a small MLP.
The ground-truth relationship is linear + Gaussian noise, so a model
that fits well will achieve R² close to the noise ceiling (~0.98).

Demonstrates:
    - Regression with MSE loss
    - Feature normalisation before training
    - R² score to measure fit quality
    - SGD converging on a well-conditioned regression problem
"""

import sys, os, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import prometheus as p

random.seed(0)

# ── Synthetic dataset ─────────────────────────────────────────────────────────
# y = 2*x1 + 3*x2 - x3 + 0.5*x4 - 1.5*x5 + N(0, 0.3)
# Features are uniform in [-1, 1] — no extra normalisation needed.
WEIGHTS = [2.0, 3.0, -1.0, 0.5, -1.5]
N       = 400

X_raw, y_raw = [], []
for _ in range(N):
    x = [random.uniform(-1.0, 1.0) for _ in range(5)]
    y = sum(w * xi for w, xi in zip(WEIGHTS, x)) + random.gauss(0, 0.3)
    X_raw.append(x)
    y_raw.append([y])

# Normalise target to zero-mean, unit-variance for stable gradients
y_mean = sum(r[0] for r in y_raw) / N
y_std  = math.sqrt(sum((r[0] - y_mean) ** 2 for r in y_raw) / N) + 1e-8
y_norm = [[(r[0] - y_mean) / y_std] for r in y_raw]

# ── Train / val split ─────────────────────────────────────────────────────────
n_train   = int(N * 0.8)
idx       = list(range(N))
random.shuffle(idx)
train_idx = idx[:n_train]
val_idx   = idx[n_train:]

def make_tensor(rows):
    flat = [v for row in rows for v in row]
    return p.Tensor([len(rows), len(rows[0])], flat)

X_train = make_tensor([X_raw[i]  for i in train_idx])
y_train = make_tensor([y_norm[i] for i in train_idx])
X_val   = make_tensor([X_raw[i]  for i in val_idx])
y_val   = make_tensor([y_norm[i] for i in val_idx])

print(f"dataset: {n_train} train  {len(val_idx)} val  |  5 features → 1 target")
print(f"noise floor R² ≈ {1 - (0.3/y_std)**2:.3f}  (limit set by noise in data)")

# ── Model ─────────────────────────────────────────────────────────────────────
layers = [
    p.Linear(5, 32),
    p.Tanh(),          # Tanh avoids the dead-neuron problem for regression
    p.Linear(32, 1),
]
model     = p.Sequential(layers)
optimizer = p.Adam(model.parameters(), lr=0.05)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n{'epoch':>6}  {'train_mse':>10}  {'val_r2':>8}")
print("-" * 34)

for epoch in range(1, 301):
    out  = model.forward(X_train)
    loss = p.mse_loss(out, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 50 == 0:
        val_out = model.forward(X_val)
        r2      = p.r2_score(val_out, y_val)
        print(f"{epoch:>6}  {loss.data[0]:>10.4f}  {r2:>8.4f}")

# ── Final evaluation ──────────────────────────────────────────────────────────
val_out  = model.forward(X_val)
val_loss = p.mse_loss(val_out, y_val)
r2       = p.r2_score(val_out, y_val)
print(f"\nfinal  val R²: {r2:.4f}  val MSE: {val_loss.data[0]:.4f}")
print("(R² = 1.0 is perfect; > 0.9 is very good)")
