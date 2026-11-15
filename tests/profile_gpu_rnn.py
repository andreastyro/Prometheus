"""Where a recurrent training step spends its GPU time, kernel by kernel:
Prometheus (its event-based kernel profiler) or PyTorch (torch.profiler), the
same model as tests/bench_gpu_lstm.py.

    python tests/profile_gpu_rnn.py prometheus|torch [--cell lstm|gru|rnn] [--top N] [--steps N]
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("lib", choices=["prometheus", "torch"])
ap.add_argument("--cell", choices=["lstm", "gru", "rnn"], default="lstm")
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--seq", type=int, default=64)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--input", type=int, default=128)
ap.add_argument("--hidden", type=int, default=256)
args = ap.parse_args()

import numpy as np

rng = np.random.default_rng(0)
S, B, I, H = args.seq, args.batch, args.input, args.hidden
x_np = rng.uniform(-1, 1, (S, B, I)).astype(np.float32)
y_np = rng.uniform(-1, 1, (S, B, H)).astype(np.float32)

if args.lib == "prometheus":
    import prometheus as p
    p.Tensor.manual_seed(0)
    model = {"lstm": p.LSTM, "gru": p.GRU, "rnn": p.RNN}[args.cell](I, H)
    model.cuda()
    x = p.Tensor([S, B, I], [float(v) for v in x_np.reshape(-1)]).cuda()
    y = p.Tensor([S, B, H], [float(v) for v in y_np.reshape(-1)]).cuda()
    opt = p.SGD(model.parameters(), 0.01)

    def step():
        opt.zero_grad()
        p.mse_loss(model.forward(x), y).backward()
        opt.step()

    for _ in range(3):
        step()
    p.cuda_synchronize()
    p.cuda_profile(True)
    p.cuda_profile_reset()
    for _ in range(args.steps):
        step()
    rows = p.cuda_profile_entries()
    p.cuda_profile(False)
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.profiler import profile, ProfilerActivity
    torch.manual_seed(0)
    dev = torch.device("cuda")
    model = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[args.cell](I, H).to(dev)
    x = torch.tensor(x_np, device=dev)
    y = torch.tensor(y_np, device=dev)
    opt = torch.optim.SGD(model.parameters(), 0.01)

    def step():
        opt.zero_grad()
        F.mse_loss(model(x)[0], y).backward()
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
        if t > 0 and e.device_type.name == "CUDA":
            rows.append((e.key, t / 1000.0, e.count))
    rows.sort(key=lambda r: -r[1])

total = sum(r[1] for r in rows)
calls = sum(r[2] for r in rows)
print("%s %s: GPU time per step %.2f ms, %.0f launches per step, %d kinds"
      % (args.lib, args.cell, total / args.steps, calls / args.steps, len(rows)))
for name, ms, n in rows[:args.top]:
    print("  %-60s %8.3f ms/step %5.1f%% %6.1f calls/step"
          % (name[:60], ms / args.steps, 100 * ms / total, n / args.steps))
