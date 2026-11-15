"""Recurrent training step: Prometheus against PyTorch, same model, same device.

    python tests/bench_gpu_lstm.py prometheus|torch [--cell lstm|gru|rnn] [--device cuda|cpu]
                                   [--seq S] [--batch B] [--input I] [--hidden H]

One recurrent layer over a [seq, batch, input] sequence, MSE of every output
against a fixed target, SGD. Forward + backward + update, median of the timed
steps. PyTorch runs its cuDNN recurrent kernels on the GPU.
"""
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("lib", choices=["prometheus", "torch"])
ap.add_argument("--cell", choices=["lstm", "gru", "rnn"], default="lstm")
ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
ap.add_argument("--seq", type=int, default=64)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--input", type=int, default=128)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--graph", action="store_true", help="Prometheus: capture the step in a CUDA Graph and time replays")
ap.add_argument("--compile-mode", default="default", help="torch.compile mode: default, max-autotune, reduce-overhead")
ap.add_argument("--compile", action="store_true", help="PyTorch only: torch.compile the model (Inductor; Triton needs compute 7.0+)")
ap.add_argument("--reset-after", action="store_true",
                help="GRU in PyTorch's form (reset_after=True); PyTorch's GRU always is")
args = ap.parse_args()

import numpy as np

rng = np.random.default_rng(0)
S, B, I, H = args.seq, args.batch, args.input, args.hidden
x_np = rng.uniform(-1, 1, (S, B, I)).astype(np.float32)
y_np = rng.uniform(-1, 1, (S, B, H)).astype(np.float32)

if args.lib == "prometheus":
    import prometheus as p
    p.Tensor.manual_seed(0)
    model = (p.GRU(I, H, reset_after=True) if args.cell == "gru" and args.reset_after
             else {"lstm": p.LSTM, "gru": p.GRU, "rnn": p.RNN}[args.cell](I, H))
    x = p.Tensor([S, B, I], [float(v) for v in x_np.reshape(-1)])
    y = p.Tensor([S, B, H], [float(v) for v in y_np.reshape(-1)])
    if args.device == "cuda":
        model.cuda()
        x, y = x.cuda(), y.cuda()
    opt = p.SGD(model.parameters(), 0.01)
    print("prometheus:", p.cuda_status() if args.device == "cuda" else p.backend())

    def step():
        opt.zero_grad()
        loss = p.mse_loss(model.forward(x), y)
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
    model = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[args.cell](I, H).to(dev)
    x = torch.tensor(x_np, device=dev)
    y = torch.tensor(y_np, device=dev)
    opt = torch.optim.SGD(model.parameters(), 0.01)
    if args.compile:      # compiled during the warm-up steps, before timing
        model = torch.compile(model, mode=None if args.compile_mode == "default" else args.compile_mode)
    print("torch:", torch.__version__, torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu")

    def step():
        opt.zero_grad()
        loss = F.mse_loss(model(x)[0], y)
        loss.backward()
        opt.step()
        return loss

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

for _ in range(3):
    step()
sync()
if args.graph and args.lib == "prometheus":
    # one captured step, replayed: the host work and the launches go
    graph = p.CUDAGraph()
    with graph.capture():
        captured_loss = step()

    def step():
        graph.replay()
        return captured_loss
    step()
    sync()
times = []
for _ in range(args.steps):
    t0 = time.perf_counter()
    loss = step()
    sync()
    times.append(time.perf_counter() - t0)
med = sorted(times)[len(times) // 2]
print("%-10s %-4s %s seq %d x batch %d, %d -> %d: median %.1f ms/step, best %.1f ms (loss %.4f)"
      % (args.lib, args.device, args.cell.upper() + (" reset_after" if args.reset_after else ""), S, B, I, H,
         med * 1e3, min(times) * 1e3, float(loss.item())))
