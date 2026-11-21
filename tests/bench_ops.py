# Single layers, forward + backward, against PyTorch on the same device:
# attention, LayerNorm, LSTM, Conv2D. The whole-model benchmarks
# (bench_gpu_*.py) show where a step's time goes in total; this shows each
# layer on its own.
#
#   python tests/bench_ops.py [--device cpu|cuda]
import argparse
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
ap.add_argument("--reps", type=int, default=10)
args = ap.parse_args()
dev = args.device

import prometheus as p  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

rng = np.random.default_rng(0)


def tm(step, sync):
    for _ in range(3):
        step()
    sync()
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        step()
        sync()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3


def P(shape):
    v = rng.standard_normal(shape).astype(np.float32)
    t = p.Tensor(list(shape), v.ravel().tolist())
    if dev == "cuda":
        t = t.cuda()
    t.requires_grad = True
    return t, v


def psync():
    if dev == "cuda":
        p.cuda_synchronize()


def tsync():
    if dev == "cuda":
        torch.cuda.synchronize()


def ours_and_theirs(name, make_p, make_t):
    pf = make_p()
    ours = tm(lambda: p.sum(p.reshape(pf(), [-1]), -1).backward(), psync)
    theirs = "--"
    if torch is not None and (dev == "cpu" or torch.cuda.is_available()):
        tf = make_t()
        theirs = "%.2f" % tm(lambda: tf().sum().backward(), tsync)
    print("| %s | %.2f | %s |" % (name, ours, theirs), flush=True)


def to_dev(m):
    if dev == "cuda":
        m.cuda() if hasattr(m, "cuda") else m.to(p.cuda)
    return m


B, S, E, H = 8, 256, 768, 12
print("%s, ms (forward + backward, median of %d)\n" % (dev, args.reps))
print("| Layer | Prometheus | PyTorch |")
print("|---|---|---|")

x, xv = P((B, S, E))
attn = p.MultiHeadAttention(E, H, True)
if dev == "cuda":
    attn.to(p.cuda)
ours_and_theirs("attention %dx%dx%d, %d heads, causal" % (B, S, E, H), lambda: lambda: attn.forward(x),
                lambda: (lambda m, tx: lambda: m(tx, tx, tx, is_causal=True,
                                                 attn_mask=torch.triu(torch.ones(S, S, dtype=torch.bool,
                                                                                 device=tx.device), 1))[0])(
                    torch.nn.MultiheadAttention(E, H, batch_first=True).to(dev),
                    torch.tensor(xv, device=dev, requires_grad=True)))

ln = p.LayerNorm(E)
if dev == "cuda":
    ln.to(p.cuda)
ours_and_theirs("LayerNorm %dx%dx%d" % (B, S, E), lambda: lambda: ln.forward(x),
                lambda: (lambda m, tx: lambda: m(tx))(torch.nn.LayerNorm(E).to(dev),
                                                      torch.tensor(xv, device=dev, requires_grad=True)))

seq, xs = P((64, 32, 128))
lstm = p.LSTM(128, 256)
if dev == "cuda":
    lstm.to(p.cuda)
ours_and_theirs("LSTM 64 x 32, 128 -> 256", lambda: lambda: lstm.forward(seq),
                lambda: (lambda m, tx: lambda: m(tx)[0])(torch.nn.LSTM(128, 256).to(dev),
                                                         torch.tensor(xs, device=dev, requires_grad=True)))

img, xi = P((16, 32, 32, 32))
conv = p.Conv2D(32, 64, 3, 1, 1)
if dev == "cuda":
    conv.to(p.cuda)
ours_and_theirs("Conv2D 16x32x32x32 -> 64, k3", lambda: lambda: conv.forward(img),
                lambda: (lambda m, tx: lambda: m(tx))(torch.nn.Conv2d(32, 64, 3, padding=1).to(dev),
                                                      torch.tensor(xi, device=dev, requires_grad=True)))
