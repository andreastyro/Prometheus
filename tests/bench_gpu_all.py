"""Every GPU benchmark against PyTorch in one run, each pair interleaved and
repeated, medians reported as one markdown table.

    python tests/bench_gpu_all.py [--rounds N]

Runs tests/bench_gpu_gpt.py, bench_gpu_cnn.py and bench_gpu_lstm.py as
subprocesses (each in a fresh process, as a user would), Prometheus and
PyTorch alternating so drift in the GPU's clocks hits both.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=2)
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


def median_ms(script, lib, extra):
    out = subprocess.run([sys.executable, os.path.join(here, script), lib] + extra,
                         capture_output=True, text=True).stdout
    m = re.search(r"median ([0-9.]+) ms", out)
    return float(m.group(1)) if m else float("nan")


rows = []
for name, script, ours, theirs in cases:
    p_ms, t_ms = [], []
    for _ in range(args.rounds):
        p_ms.append(median_ms(script, "prometheus", ours))
        if theirs is not None:
            t_ms.append(median_ms(script, "torch", theirs))
    p = statistics.median(p_ms)
    t = statistics.median(t_ms) if t_ms else None
    rows.append((name, p, t))
    print("%-34s prometheus %6.1f ms   torch %s" % (name, p, "%6.1f ms" % t if t else "    --"), flush=True)

print("\n| Training step | Prometheus | PyTorch | |\n|---|---|---|---|")
for name, p, t in rows:
    if t:
        ratio = t / p
        note = "%.2fx faster" % ratio if ratio >= 1.0 else "%.2fx slower" % (1.0 / ratio)
        print("| %s | %.1f ms | %.1f ms | %s |" % (name, p, t, note))
    else:
        print("| %s | %.1f ms | -- | |" % (name, p))
