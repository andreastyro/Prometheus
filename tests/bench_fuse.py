# p.fuse against the same function as separate ops, and against PyTorch
# (eager and torch.compile), forward + backward, on one GPU.
#
#   python tests/bench_fuse.py
import time

import numpy as np
import prometheus as p

try:
    import torch
except ImportError:
    torch = None


def swish_mse(x, t):
    y = x * p.fx.sigmoid(x) - t
    return p.fx.mean(y * y)


def block(x, g, b, w1, b1, w2, b2):
    h = p.fx.layernorm(x, g, b)
    h = p.fx.gelu(h @ w1 + b1)
    return h @ w2 + b2 + x


def t_swish_mse(x, t):
    y = x * torch.sigmoid(x) - t
    return (y * y).mean(-1, keepdim=True)


def t_block(x, g, b, w1, b1, w2, b2):
    h = torch.nn.functional.layer_norm(x, (x.shape[-1],), g, b)
    h = torch.nn.functional.gelu(h @ w1 + b1, approximate="tanh")
    return h @ w2 + b2 + x


cases = [("swish + MSE, 4096 x 1024", swish_mse, t_swish_mse, [(4096, 1024), (1024,)]),
         ("LayerNorm + MLP block, 2048 x 768", block, t_block,
          [(2048, 768), (768,), (768,), (768, 3072), (3072,), (3072, 768), (768,)])]


def timeit(step, sync, n=30):
    for _ in range(5):
        step()
    sync()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        step()
        sync()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3


rng = np.random.default_rng(0)
print("| Function (fwd + bwd) | ops | p.fuse | launches ops -> fuse | torch eager | torch.compile |")
print("|---|---|---|---|---|---|")
for name, fn, tfn, shapes in cases:
    vals = [(rng.standard_normal(s) * 0.1).astype(np.float32) for s in shapes]
    ts = [p.Tensor(list(v.shape), v.reshape(-1).tolist()).cuda() for v in vals]
    for t in ts:
        t.requires_grad = True
    fused = p.fuse(fn)

    def run(f):
        def step():
            y = f(*ts)
            p.sum(p.reshape(y, [-1]), -1).backward()
        return step
    counts = []
    for f in (fn, fused):
        run(f)()
        p.cuda_synchronize()
        p.cuda_reset_launch_counts()
        run(f)()
        p.cuda_synchronize()
        c = p.cuda_launch_counts()
        counts.append(c["kernels"] + c["library"])
    ops_ms = timeit(run(fn), p.cuda_synchronize)
    fuse_ms = timeit(run(fused), p.cuda_synchronize)
    te = tc = "--"
    if torch is not None and torch.cuda.is_available():
        tt = [torch.tensor(v, device="cuda", requires_grad=True) for v in vals]

        def trun(f):
            def step():
                f(*tt).sum().backward()
            return step
        te = "%.3f" % timeit(trun(tfn), torch.cuda.synchronize)
        try:
            tc = "%.3f" % timeit(trun(torch.compile(tfn)), torch.cuda.synchronize)
        except Exception as e:  # noqa: BLE001
            tc = "failed (%s)" % type(e).__name__
    print("| %s | %.3f | %.3f | %d -> %d | %s | %s |" % (name, ops_ms, fuse_ms, counts[0], counts[1], te, tc),
          flush=True)
