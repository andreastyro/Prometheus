"""Build Prometheus on a rented GPU with Modal and run the GPU suites and the
benchmarks against PyTorch (eager and torch.compile) there.

    pip install modal && modal token new            # once
    modal run tests/modal_gpu.py                    # A100, tests + benchmarks
    modal run tests/modal_gpu.py --gpu H100
    modal run tests/modal_gpu.py --gpu T4 --no-bench
    modal run tests/modal_gpu.py --rounds 3 --no-compile

Each run starts a fresh Linux container with the GPU, uploads this checkout
(source only), builds the Python module and the C++ GPU suite, runs both
suites, then tests/bench_gpu_all.py. Billed per second while it runs.
"""
import os

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# torch brings the CUDA 12 runtime pieces Prometheus loads at runtime
# (nvidia-cuda-nvrtc-cu12, nvidia-cublas-cu12, nvidia-cudnn-cu12) as its own
# dependencies, so the same packages serve both libraries.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("g++", "make")
    .pip_install("numpy", "pybind11", "setuptools", "wheel", "torch")
    .add_local_dir(
        ROOT, "/root/prometheus",
        ignore=["build", "build/**", ".git", ".git/**", "Testing", "Testing/**", "**/__pycache__",
                "**/*.pyd", "**/*.so", "**/*.egg-info", "*.egg-info/**", "PLAN.md", "CUDA_PLAN.md",
                "ROADMAP.md", ".vscode", ".vscode/**"],
    )
)

app = modal.App("prometheus-gpu", image=image)


@app.function(gpu="A100", timeout=3600)
def run(bench: bool = True, rounds: int = 2, compile_: bool = True) -> str:
    import glob
    import subprocess
    import sys
    import time

    os.chdir("/root/prometheus")
    site = subprocess.check_output([sys.executable, "-c", "import site; print(site.getsitepackages()[0])"],
                                   text=True).strip()
    # NVRTC / cuBLAS / cuDNN from the pip packages, for the C++ suite (the
    # Python module finds them by itself)
    libs = sorted(glob.glob(os.path.join(site, "nvidia", "*", "lib")))
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")]),
               PROMETHEUS_USE_OPENMP="1")
    log = []

    def step(title, cmd):
        t0 = time.time()
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        text = (out.stdout + out.stderr).strip()
        head = "== %s (%.0f s, exit %d)" % (title, time.time() - t0, out.returncode)
        print(head, flush=True)
        log.append(head)
        keep = [l for l in text.splitlines() if any(k in l for k in (
            "checks", "FAIL", "failed", "error", "Error", "compiled", "built", "PASS", "|", "prometheus ",
            "torch ", "GPU", "cuda", "cuDNN", "median", "compute"))]
        fails = [l for l in text.splitlines() if "FAIL" in l]
        body = "\n".join(keep[-60:]) if out.returncode == 0 else "\n".join(fails + text.splitlines()[-25:])
        print(body, flush=True)
        log.append(body)
        return out.returncode

    step("GPU", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    step("torch", [sys.executable, "-c", "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0),"
                                         " 'cuDNN', torch.backends.cudnn.version())"])
    if step("build the Python module", [sys.executable, "-m", "pip", "install", ".", "--no-build-isolation", "-v"]):
        return "\n".join(log)
    step("Prometheus sees", [sys.executable, "-c", "import prometheus as p; print(p.cuda_status());"
                                                   " print(p.cuda_cudnn_status())"])
    step("C++ GPU suite", [sys.executable, "tests/build_gpu_tests.py", "--run"])
    step("Python GPU suite", [sys.executable, "tests/test_cuda.py"])
    if bench:
        cmd = [sys.executable, "tests/bench_gpu_all.py", "--rounds", str(rounds)]
        if compile_:
            cmd.append("--compile")
        step("benchmarks against PyTorch", cmd)
    return "\n".join(log)


@app.local_entrypoint()
def main(gpu: str = "A100", bench: bool = True, rounds: int = 2, compile: bool = True):
    print("running on", gpu)
    out = run.with_options(gpu=gpu).remote(bench=bench, rounds=rounds, compile_=compile)
    stamp = gpu.replace(":", "_")
    path = os.path.join(ROOT, "build", "modal_%s.log" % stamp)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print("\nlog saved to", path)
