# The recurrent layers with and without the single-launch recurrence
# (PROMETHEUS_PERSISTENT_RNN), against PyTorch's cuDNN, on one GPU. Each run
# is its own process, as the switch is read once.
import os
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
cases = [("LSTM", ["--cell", "lstm"]), ("GRU reset_after", ["--cell", "gru", "--reset-after"]),
         ("RNN", ["--cell", "rnn"])]
extra = sys.argv[1:]


def median(lib, args, env_value=None):
    env = dict(os.environ)
    if env_value is not None:
        env["PROMETHEUS_PERSISTENT_RNN"] = env_value
    out = subprocess.run([sys.executable, os.path.join(here, "bench_gpu_lstm.py"), lib] + args + extra,
                         env=env, capture_output=True, text=True)
    text = out.stdout + out.stderr
    m = re.search(r"median ([\d.]+) ms", text)
    launches = re.search(r"launches per step: (\d+) kernels, (\d+) cuBLAS", text)
    if not m:
        return "failed: " + text.strip().splitlines()[-1] if text.strip() else "failed"
    return m.group(1) + (" (%s+%s launches)" % launches.groups() if launches else "")


print("| Layer | per step | one launch | PyTorch |")
print("|---|---|---|---|")
for name, args in cases:
    print("| %s | %s | %s | %s |" % (name, median("prometheus", args, "0"), median("prometheus", args, "1"),
                                     median("torch", args)), flush=True)
