"""
XOR example — the simplest possible neural network problem.

XOR (exclusive-or) returns 1 when exactly one input is 1, otherwise 0:
    0 XOR 0 = 0
    0 XOR 1 = 1
    1 XOR 0 = 1
    1 XOR 1 = 0

A single linear layer can't solve XOR because the two classes (0 and 1)
are not linearly separable — you can't draw one straight line to separate them.
Adding a hidden layer with a non-linear activation (ReLU) gives the network
enough capacity to learn the curved boundary.

This example shows the basic training loop:
    forward pass -> compute loss -> backward -> optimizer step
"""

import sys
sys.path.insert(0, "c:/Users/andre/Desktop/Projects/prometheus")

import prometheus as p

# XOR truth table — all 4 possible input combinations
# Shape [4, 2]: 4 samples, each with 2 input features
X = p.Tensor([4, 2], [
    0.0, 0.0,  # 0 XOR 0
    0.0, 1.0,  # 0 XOR 1
    1.0, 0.0,  # 1 XOR 0
    1.0, 1.0,  # 1 XOR 1
])

# Expected output for each input pair
# Shape [4, 1]: one label per sample
y = p.Tensor([4, 1], [
    0.0,  # 0 XOR 0 = 0
    1.0,  # 0 XOR 1 = 1
    1.0,  # 1 XOR 0 = 1
    0.0,  # 1 XOR 1 = 0
])

X.requires_grad = True  # track gradients through the input

# Model: 2 inputs -> 8 hidden neurons (ReLU) -> 1 output (Sigmoid)
# Sigmoid squashes the output to (0, 1) so it can be compared to 0/1 labels
model = p.Sequential([
    p.Linear(2, 8),   # learn a weighted combination of the two inputs
    p.ReLU(),         # introduce non-linearity so XOR is solvable
    p.Linear(8, 1),   # compress hidden features down to a single prediction
    p.Sigmoid(),      # squash to (0, 1) probability range
])

optimizer = p.Adam(model.parameters(), lr=0.01)

# Training loop — 1000 gradient updates
for epoch in range(1000):
    out  = model.forward(X)          # forward pass: predict outputs
    loss = p.mse_loss(out, y)        # mean squared error between pred and truth
    loss.backward()                  # compute gradients for all parameters
    optimizer.step()                 # update weights in the direction that reduces loss
    optimizer.zero_grad()            # clear gradients before the next iteration

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch+1:4d}  loss {loss.data[0]:.4f}")

# Final predictions — should be close to 0, 1, 1, 0
print("\npredictions:")
out = model.forward(X)
for i in range(4):
    x0   = X.data[i * 2]
    x1   = X.data[i * 2 + 1]
    pred = out.data[i]
    print(f"  {int(x0)} XOR {int(x1)} = {pred:.4f}  (expected {int(y.data[i])})")
