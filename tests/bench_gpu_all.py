"""Every GPU benchmark against PyTorch in one run, each pair interleaved and
repeated, medians reported as one markdown table.

    python tests/bench_gpu_all.py [--rounds N] [--compile]

Runs tests/bench_gpu_gpt.py, bench_gpu_cnn.py and bench_gpu_lstm.py as
subprocesses (each in a fresh process, as a user would), Prometheus and
PyTorch alternating so drift in the GPU's clocks hits both. --compile adds
PyTorch with torch.compile (Inductor's GPU backend is Triton: Linux, compute
capability 7.0 or newer).
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--compile", action="store_true", help="also PyTorch with torch.compile")
ap.add_argument("--wide", action="store_true",
                help="the large-model set instead: GPT 12 x 768 in float32, TF32 and bf16 autocast "
                     "(torch.compile in max-autotune with --compile)")
ap.add_argument("--graphs", action="store_true",
                help="Prometheus with each step replayed from a CUDA Graph; torch.compile in reduce-overhead mode "
                     "(PyTorch's CUDA Graphs) with --compile")
args = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))

cases = [
    ("GPT 4 x 256, batch 8 x 128", "bench_gpu_gpt.py", [], []),
    ("GPT, clip_grad_norm every step", "bench_gpu_gpt.py", ["--clip"], ["--clip"]),
    ("GPT with RoPE", "bench_gpu_gpt.py", ["--rope"], None),
    ("CNN, batch 64 of 32x32", "bench_gpu_cnn.py", [], []),
    ("LSTM 64 x 32, 128 -> 256", "bench_gpu_lstm.py", ["--cell", "lstm"], ["--cell", "lstm"]),
    ("GRU (reset_after=True)", "bench_gpu_lstm.py", ["--cell", "gru", "--reset-after"], ["--cell", "gru"]),
    ("GRU (default form)", "bench_gpu_lstm.py", ["--cell", "gru"], None),
    ("RNN", "bench_gpu_lstm.py", ["--cell", "rnn"], ["--cell", "rnn"]),
]


if args.wide:
    big = ["--layers", "12", "--embed", "768", "--heads", "12", "--seq", "256", "--batch", "8", "--steps", "10"]
    cases = [
        ("GPT 12 x 768, seq 256, float32", "bench_gpu_gpt.py", big, big),
        ("GPT 12 x 768, TF32 matmuls", "bench_gpu_gpt.py", big + ["--tf32"], big + ["--tf32"]),
        ("GPT 12 x 768, bf16 autocast", "bench_gpu_gpt.py", big + ["--amp", "bf16"], big + ["--amp", "bf16"]),
        ("GPT 4 x 256, bf16 autocast", "bench_gpu_gpt.py", ["--amp", "bf16"], ["--amp", "bf16"]),
    ]
if args.graphs:
    cases = [(name + ", CUDA Graph", script, ours + ["--graph"], theirs)
             for name, script, ours, theirs in cases if theirs is not None and "--clip" not in ours]


def median_ms(script, lib, extra):
    out = subprocess.run([sys.executable, os.path.join(here, script), lib] + extra,
                         capture_output=True, text=True)
    m = re.search(r"median ([0-9.]+) ms", out.stdout)
    if not m:
        tail = (out.stderr or out.stdout).strip().splitlines()[-1:] or ["no output"]
        print("  %s %s %s failed: %s" % (script, lib, " ".join(extra), tail[0]), flush=True)
        return float("nan")
    return float(m.group(1))


def med(v):
    v = [x for x in v if x == x]
    return statistics.median(v) if v else None


rows = []
for name, script, ours, theirs in cases:
    p_ms, t_ms, c_ms = [], [], []
    for _ in range(args.rounds):
        p_ms.append(median_ms(script, "prometheus", ours))
        if theirs is not None:
            t_ms.append(median_ms(script, "torch", theirs))
            if args.compile:
                mode = (["--compile-mode", "max-autotune"] if args.wide else
                        ["--compile-mode", "reduce-overhead"] if args.graphs else [])
                c_ms.append(median_ms(script, "torch", theirs + ["--compile"] + mode))
    p, t, c = med(p_ms), med(t_ms), med(c_ms)
    rows.append((name, p, t, c))
    fmt = lambda v: "%6.1f ms" % v if v else "    --"
    print("%-34s prometheus %s   torch %s   compiled %s" % (name, fmt(p), fmt(t), fmt(c)), flush=True)


def ms(v):
    return "%.1f ms" % v if v else "--"


def vs(p, t):
    if not p or not t:
        return "--"
    r = t / p
    return "%.2fx faster" % r if r >= 1.0 else "%.2fx slower" % (1.0 / r)


print()
if args.compile:
    print("| Training step | Prometheus | PyTorch | torch.compile | vs PyTorch | vs torch.compile |")
    print("|---|---|---|---|---|---|")
    for name, p, t, c in rows:
        print("| %s | %s | %s | %s | %s | %s |" % (name, ms(p), ms(t), ms(c), vs(p, t), vs(p, c)))
else:
    print("| Training step | Prometheus | PyTorch | |")
    print("|---|---|---|---|")
    for name, p, t, c in rows:
        print("| %s | %s | %s | %s |" % (name, ms(p), ms(t), vs(p, t)))
