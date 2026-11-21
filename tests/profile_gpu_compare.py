# The large GPT step under bf16, profiled kernel by kernel: Prometheus,
# PyTorch eager and torch.compile (max-autotune), one after the other.
#
#   python tests/profile_gpu_compare.py [extra args for profile_gpu_gpt.py]
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
size = ["--layers", "12", "--embed", "768", "--heads", "12", "--seq", "256", "--batch", "8",
        "--amp", "bf16", "--top", "30"] + sys.argv[1:]
for label, args, env in [("Prometheus", ["prometheus"], {}),
                         ("Prometheus, LayerNorm split", ["prometheus"], {"PROMETHEUS_LN_SPLIT": "1"}),
                         ("PyTorch eager", ["torch"], {})]:
    print("\n==== %s ====" % label, flush=True)
    out = subprocess.run([sys.executable, os.path.join(here, "profile_gpu_gpt.py")] + args + size,
                         capture_output=True, text=True, env=dict(os.environ, **env))
    print((out.stdout + out.stderr).strip()[-6000:], flush=True)
