"""
Prometheus ML Library — full demo.

Trains a 3-layer MLP to classify a synthetic spiral dataset.
The spiral has two interleaved arms — it's a classic non-linear problem
that a single linear layer cannot solve.

Demonstrates:
    - Building a model with Sequential
    - Binary cross-entropy loss + Adam optimiser
    - Learning rate scheduling (StepLR)
    - Classification metrics (accuracy, precision, recall, F1)
    - Mixed precision (half()/GradScaler)
    - ONNX export
"""

import sys, math, random
sys.path.insert(0, "c:/Users/andre/Desktop/Projects/prometheus")

import prometheus as p

# ─────────────────────────────────────────────
# 1. Generate spiral dataset (2 classes, 2 features)
# ─────────────────────────────────────────────
random.seed(42)

def make_spiral(n=200):
    """
    Generate a two-class spiral dataset.

    Each class forms a spiral arm. The two arms interleave, making this
    impossible to separate with a straight line — the model must learn
    curved decision boundaries.

    Returns:
        X: list of [x, y] coordinate pairs
        y: list of [label] where label is 0 or 1
    """
    X, y = [], []
    for cls in range(2):
        for i in range(n):
            angle = i / n * 4 * math.pi + cls * math.pi  # spiral angle
            r     = i / n                                  # radius grows with i
            noise = (random.random() - 0.5) * 0.2        # small jitter
            X.append([r * math.cos(angle) + noise,
                       r * math.sin(angle) + noise])
            y.append([float(cls)])
    return X, y

X_list, y_list = make_spiral(200)
n_total = len(X_list)
n_train = int(n_total * 0.8)  # 80% train, 20% validation

# Shuffle indices so train/val split is random (not first-half / second-half)
random.seed(0)
idx = list(range(n_total))
random.shuffle(idx)

def make_tensor(rows):
    """Convert a list of rows into a prometheus Tensor."""
    flat = [v for row in rows for v in row]
    return p.Tensor([len(rows), len(rows[0])], flat)

X_train = make_tensor([X_list[i] for i in idx[:n_train]])
y_train = make_tensor([y_list[i] for i in idx[:n_train]])
X_val   = make_tensor([X_list[i] for i in idx[n_train:]])
y_val   = make_tensor([y_list[i] for i in idx[n_train:]])

X_train.requires_grad = True  # needed so gradients flow through the input

print(f"dataset: {n_train} train  {n_total - n_train} val")

# ─────────────────────────────────────────────
# 2. Build model
# ─────────────────────────────────────────────
# 2 inputs (x, y coordinates) -> 32 hidden -> 32 hidden -> 1 output (probability)
layers = [
    p.Linear(2, 32),   # learn to separate the two spiral arms
    p.ReLU(),
    p.Linear(32, 32),  # second hidden layer for more capacity
    p.ReLU(),
    p.Linear(32, 1),   # compress to a single logit
    p.Sigmoid(),       # convert logit to probability in (0, 1)
]
model = p.Sequential(layers)

# ─────────────────────────────────────────────
# 3. Optimiser + LR scheduler
# ─────────────────────────────────────────────
optimizer = p.Adam(model.parameters(), lr=0.01)

# Halve the learning rate every 50 epochs — start bold, finish precise
scheduler = p.StepLR(base_lr=0.01, step_size=50, gamma=0.5)

# ─────────────────────────────────────────────
# 4. Training loop
# ─────────────────────────────────────────────
print("\ntraining...")
print(f"{'epoch':>6}  {'loss':>8}  {'val_loss':>9}  {'lr':>8}")
print("-" * 40)

for epoch in range(1, 201):
    # Clip predictions to (1e-7, 1-1e-7) to avoid log(0) in BCE loss
    out  = p.clip(model.forward(X_train), 1e-7, 1 - 1e-7)
    loss = p.bce_loss(out, y_train)   # binary cross-entropy
    loss.backward()                    # compute gradients
    optimizer.step()                   # update weights
    optimizer.zero_grad()              # clear gradients for next iteration

    # Decay the learning rate once per epoch
    optimizer.lr = scheduler.step()

    if epoch % 20 == 0:
        val_out  = p.clip(model.forward(X_val), 1e-7, 1 - 1e-7)
        val_loss = p.bce_loss(val_out, y_val)
        print(f"{epoch:>6}  {loss.data[0]:>8.4f}  {val_loss.data[0]:>9.4f}  {optimizer.lr:>8.6f}")

# ─────────────────────────────────────────────
# 5. Metrics
# ─────────────────────────────────────────────
print("\n--- metrics ---")
val_out = model.forward(X_val)

# Round probabilities to hard predictions: >= 0.5 -> 1, < 0.5 -> 0
pred_rounded = p.Tensor(val_out.shape, [round(v) for v in val_out.data])

acc  = p.accuracy( pred_rounded, y_val)
prec = p.precision(pred_rounded, y_val)
rec  = p.recall(   pred_rounded, y_val)
f1   = p.f1_score( pred_rounded, y_val)
print(f"accuracy:  {acc:.4f}")
print(f"precision: {prec:.4f}")
print(f"recall:    {rec:.4f}")
print(f"f1 score:  {f1:.4f}")

# ─────────────────────────────────────────────
# 6. Mixed precision demo
# ─────────────────────────────────────────────
# Shows how float16 trades off memory/speed for a small loss in precision.
# In a real mixed-precision workflow you'd keep the model in float32 but
# run forward/backward in float16. GradScaler handles the gradient scaling.
print("\n--- mixed precision ---")
w = model.parameters()[0]
w_half = w.half()   # quantise to float16 bit representation
print(f"weight[0] float32: {w.data[0]:.8f}")
print(f"weight[0] float16: {w_half.data[0]:.8f}  (precision loss visible)")

# GradScaler: multiply loss by a large scale before backward so small
# gradients don't underflow in float16, then divide back before optimizer step
scaler = p.GradScaler(init_scale=128.0)
loss2  = p.bce_loss(model.forward(X_train), y_train)
scaled = scaler.scale_loss(loss2)       # loss * scale
print(f"loss: {loss2.data[0]:.4f}  scaled: {scaled.data[0]:.4f}")
scaled.backward()
clean = scaler.unscale(model.parameters())  # divide grads back down
print(f"gradients clean (no overflow): {clean}")
scaler.update(not clean)  # adjust scale for next step
optimizer.zero_grad()

# ─────────────────────────────────────────────
# 7. ONNX export
# ─────────────────────────────────────────────
# Saves the trained model in a standard format that can be loaded by
# ONNX Runtime, TensorFlow, or deployed on edge devices without Python.
print("\n--- ONNX export ---")
from prometheus_onnx import save_onnx
save_onnx(layers, input_shape=[1, 2], path="examples/spiral_model.onnx")

print("\ndone.")
