"""A longer training run: memory must stop growing and the loss must keep
falling. Catches leaks (a graph or buffer kept alive per step) and drift that
the short tests cannot see. CPU always, the GPU too when there is one.

    python tests/test_long_training.py [--steps N]
"""
import argparse
import os
import sys

import numpy as np
import prometheus as p

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=300)
args = ap.parse_args()

CHECKS = [0]
FAILURES = []


def check(name, ok, detail=""):
    CHECKS[0] += 1
    print("  %-64s %s %s" % (name, "ok" if ok else "** FAIL **", detail))
    if not ok:
        FAILURES.append(name)


def rss_mb():
    """This process's resident memory, MB (Linux /proc, Windows psapi; 0 if unknown)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 2 ** 20
    return 0.0


def train(dev):
    print("\n=== %d training steps, %s ===\n" % (args.steps, dev))
    rng = np.random.default_rng(0)
    V, S, B = 64, 32, 4
    # a learnable stream: a fixed random sequence of 97 tokens, repeated
    text = np.tile(rng.integers(0, V, 97), 43)
    p.Tensor.manual_seed(0)
    model = p.GPT(V, S, 64, 4, 2)
    if dev == "cuda":
        model.cuda()
    opt = p.AdamW(model.parameters(), 3e-3)
    losses, mem = [], {}
    for step in range(args.steps):
        i = (step * 37) % (len(text) - S - 1)
        x = p.Tensor([B, S], [float(text[(i + b * 7 + t) % len(text)]) for b in range(B) for t in range(S)], p.int64)
        y = p.Tensor([B * S], [float(text[(i + b * 7 + t + 1) % len(text)]) for b in range(B) for t in range(S)],
                     p.int64)
        if dev == "cuda":
            x, y = x.cuda(), y.cuda()
        opt.zero_grad()
        loss = p.cross_entropy_sparse_seq(model.forward(x), y)
        loss.backward()
        p.clip_grad_norm(model.parameters(), 1.0, return_norm=False)
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            losses.append(loss.item())
        if step in (args.steps // 4, args.steps - 1):
            mem[step] = p.cuda_memory_stats()["in_use"] / 2 ** 20 if dev == "cuda" else rss_mb()
    first, last = mem[args.steps // 4], mem[args.steps - 1]
    check("loss falls over the run (%.3f -> %.3f)" % (losses[0], losses[-1]), losses[-1] < 0.8 * losses[0])
    check("loss stays finite", all(np.isfinite(losses)))
    grow = (last - first) / max(first, 1.0)
    unit = "MB held by tensors" if dev == "cuda" else "MB resident"
    check("memory flat from step %d to %d (%.1f -> %.1f %s)" % (args.steps // 4, args.steps - 1, first, last, unit),
          first == 0.0 or grow < 0.10)


def main():
    print("long training run")
    train("cpu")
    if p.cuda_available():
        train("cuda")
    print("\n%d checks, %d failed" % (CHECKS[0], len(FAILURES)))
    print("PASS" if not FAILURES else "FAIL")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
