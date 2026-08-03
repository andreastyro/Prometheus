"""
MNIST-style digit classifier using convolutional layers.

Same task as mnist_linear.py but solved with a CNN:
Conv -> Pool -> Conv -> Pool -> Flatten -> Linear.

This architecture is much better suited to image data because it learns
spatial patterns (edges, curves) that are position-invariant.

Demonstrates:
    - Conv2D, MaxPool2D, Flatten
    - Multi-class classification pipeline
    - How spatial feature maps shrink through the network
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import prometheus as p

random.seed(7)

# ── Synthetic 28×28 image dataset ─────────────────────────────────────────────
# Each class has a bright pixel region at a class-specific location.
N_CLASSES = 10
N_PER_CLS = 10        # small — conv is O(H*W*k*k) per filter
H, W      = 28, 28

X_all, y_all = [], []
for cls in range(N_CLASSES):
    row_signal = (cls // 5) * 14       # top or bottom half
    col_signal = (cls % 5)  * 5        # five horizontal bands
    for _ in range(N_PER_CLS):
        img = [random.gauss(0.0, 0.3) for _ in range(H * W)]
        for dr in range(4):
            for dc in range(4):
                r = min(row_signal + dr, H - 1)
                c = min(col_signal + dc, W - 1)
                img[r * W + c] += 1.5  # bright 4×4 patch
        X_all.append(img)
        one_hot = [0.0] * N_CLASSES
        one_hot[cls] = 1.0
        y_all.append(one_hot)

n_total = len(X_all)
n_train = int(n_total * 0.8)

idx = list(range(n_total))
random.shuffle(idx)
train_idx = idx[:n_train]
val_idx   = idx[n_train:]

def make_img_tensor(indices):
    # Shape: [batch, 1 channel, H, W]
    batch = len(indices)
    flat  = [v for i in indices for v in X_all[i]]
    return p.Tensor([batch, 1, H, W], flat)

def make_label_tensor(indices):
    flat = [v for i in indices for v in y_all[i]]
    return p.Tensor([len(indices), N_CLASSES], flat)

X_train = make_img_tensor(train_idx)
y_train = make_label_tensor(train_idx)
X_val   = make_img_tensor(val_idx)
y_val   = make_label_tensor(val_idx)

print(f"dataset: {n_train} train  {len(val_idx)} val  |  1×{H}×{W} images  {N_CLASSES} classes")

# ── CNN architecture ──────────────────────────────────────────────────────────
# Conv(1→8, k=3, pad=1): 28×28 → 28×28
# MaxPool(2):             28×28 → 14×14
# Conv(8→16, k=3, pad=1):14×14 → 14×14
# MaxPool(2):             14×14 → 7×7
# Flatten:                       → 16*7*7 = 784
# Linear(784→64)→ReLU→Linear(64→10)→Softmax
layers = [
    p.Conv2D(1, 8, 3, 1, 1, "xavier"),   # in_ch, out_ch, kernel, stride, padding, init
    p.ReLU(),
    p.MaxPool2D(2),
    p.Conv2D(8, 16, 3, 1, 1, "xavier"),
    p.ReLU(),
    p.MaxPool2D(2),
    p.Flatten(),
    p.Linear(16 * 7 * 7, 64, "xavier"),
    p.ReLU(),
    p.Linear(64, N_CLASSES, "xavier"),
    p.Softmax(),
]
model = p.Sequential(layers)

optimizer = p.Adam(model.parameters(), lr=0.001)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n{'epoch':>6}  {'loss':>8}")
print("-" * 20)

for epoch in range(1, 31):
    out  = model.forward(X_train)
    out  = p.clip(out, 1e-7, 1.0)
    loss = p.cross_entropy_loss(out, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 5 == 0:
        print(f"{epoch:>6}  {loss.data[0]:>8.4f}")

# ── Final accuracy ────────────────────────────────────────────────────────────
val_out = model.forward(X_val)
preds   = p.argmax(val_out)
labels  = p.argmax(y_val)
correct = sum(1 for i in range(len(val_idx))
              if int(preds.data[i]) == int(labels.data[i]))
print(f"\nfinal val accuracy: {correct / len(val_idx):.3f}")
