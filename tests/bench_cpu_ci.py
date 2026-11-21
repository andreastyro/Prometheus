# CPU training steps against PyTorch on the same machine, failing on a
# slowdown: each model's time over PyTorch's must stay under a limit. Ratios,
# not absolute times, so the varying speed of CI machines cancels out. The
# limits are today's ratios on a Ryzen AI 7 350 (GPT ~1.0, CNN ~1.07, LSTM
# ~1.45) with room for the noise of a shared runner.
#
#   python tests/bench_cpu_ci.py
import os
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
cases = [
    ("GPT 4 x 256", "bench_gpu_gpt.py", ["--steps", "5"], 1.35),
    ("CNN, batch 64", "bench_gpu_cnn.py", ["--steps", "5"], 1.45),
    ("LSTM 64 x 32", "bench_gpu_lstm.py", ["--cell", "lstm", "--steps", "5"], 1.9),
]


def median(script, lib, args):
    out = subprocess.run([sys.executable, os.path.join(here, script), lib, "--device", "cpu"] + args,
                         capture_output=True, text=True, env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE"))
    m = re.search(r"median ([\d.]+) ms", out.stdout + out.stderr)
    if not m:
        sys.exit("bench_cpu_ci: %s %s failed:\n%s" % (script, lib, (out.stdout + out.stderr)[-2000:]))
    return float(m.group(1))


failed = []
print("| Model | Prometheus | PyTorch | ratio | limit |")
print("|---|---|---|---|---|")
for name, script, args, limit in cases:
    # interleaved, best of two each: one slow moment on a shared runner is not a regression
    ours = min(median(script, "prometheus", args) for _ in range(2))
    theirs = min(median(script, "torch", args) for _ in range(2))
    ratio = ours / theirs
    print("| %s | %.1f ms | %.1f ms | %.2f | %.2f |" % (name, ours, theirs, ratio, limit), flush=True)
    if ratio > limit:
        failed.append(name)
if failed:
    sys.exit("bench_cpu_ci: slower than the limit against PyTorch: " + ", ".join(failed))
print("PASS")
