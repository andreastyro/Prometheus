"""CNN training step: Prometheus against PyTorch, same model, same device.

    python tests/bench_gpu_cnn.py prometheus|torch [--device cuda|cpu] [--batch B]

conv(3->32) relu maxpool, conv(32->64) relu maxpool, conv(64->128) relu
avgpool, flatten, linear(->10); 32x32 inputs; cross-entropy; SGD with momentum.
Forward + backward + update, median of the timed steps.
"""
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("lib", choices=["prometheus", "torch"])
ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--steps", type=int, default=20)
args = ap.parse_args()

import numpy as np

rng = np.random.default_rng(0)
x_np = rng.uniform(-1, 1, (args.batch, 3, 32, 32)).astype(np.float32)
y_np = rng.integers(0, 10, args.batch)

if args.lib == "prometheus":
    import prometheus as p
    p.Tensor.manual_seed(0)
    model = p.Sequential([
        p.Conv2D(3, 32, 3, 1, 1), p.ReLU(), p.MaxPool2D(2),
        p.Conv2D(32, 64, 3, 1, 1), p.ReLU(), p.MaxPool2D(2),
        p.Conv2D(64, 128, 3, 1, 1), p.ReLU(), p.AvgPool2D(2),
        p.Flatten(), p.Linear(128 * 4 * 4, 10)])
    if args.device == "cuda":
        model.cuda()
    opt = p.SGD(model.parameters(), 0.01, 0.9)
    x = p.Tensor(list(x_np.shape), [float(v) for v in x_np.reshape(-1)])
    if args.device == "cuda":
        x = x.cuda()
    targets = [int(t) for t in y_np]
    print("prometheus:", p.cuda_status() if args.device == "cuda" else p.backend())

    def step():
        opt.zero_grad()
        loss = p.cross_entropy_sparse_seq(model.forward(x), targets)
        loss.backward()
        opt.step()
        return loss

    def sync():
        if args.device == "cuda":
            p.cuda_synchronize()
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = torch.device(args.device)
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.AvgPool2d(2),
        nn.Flatten(), nn.Linear(128 * 4 * 4, 10)).to(dev)
    opt = torch.optim.SGD(model.parameters(), 0.01, 0.9)
    x = torch.tensor(x_np, device=dev)
    y = torch.tensor(y_np, device=dev)
    print("torch:", torch.__version__, torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu")

    def step():
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        return loss

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

for _ in range(3):
    step()
sync()
times = []
for _ in range(args.steps):
    t0 = time.perf_counter()
    loss = step()
    sync()
    times.append(time.perf_counter() - t0)
med = sorted(times)[len(times) // 2]
print("%-10s %-4s CNN batch %d: median %.1f ms/step, best %.1f ms -> %.0f images/s (loss %.3f)"
      % (args.lib, args.device, args.batch, med * 1e3, min(times) * 1e3, args.batch / med, float(loss.item())))
