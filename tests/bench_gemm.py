# CPU matmul throughput (GFLOP/s) by size and thread count: the built-in
# packed kernel, the BLAS library (MKL / OpenBLAS) and PyTorch. Each
# configuration in its own process (the thread count is read at startup).
#
#   python tests/bench_gemm.py
import os
import subprocess
import sys

SIZES = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (32, 1024, 256), (256, 256, 256), (512, 512, 512),
         (1024, 768, 3072), (2048, 2048, 2048)]
THREADS = [1, 4, 16]

CHILD = r'''
import sys, time, numpy as np
lib, sizes = sys.argv[1], eval(sys.argv[2])
rng = np.random.default_rng(0)
out = []
for (m, k, n) in sizes:
    a = rng.standard_normal((m, k)).astype(np.float32); b = rng.standard_normal((k, n)).astype(np.float32)
    if lib == "torch":
        import torch
        A, B = torch.tensor(a), torch.tensor(b)
        f = lambda: A @ B
    else:
        import prometheus as p
        A, B = p.Tensor([m, k], a.ravel().tolist()), p.Tensor([k, n], b.ravel().tolist())
        f = lambda: p.matmul(A, B)
    f()
    reps = max(3, int(2e8 / (2 * m * k * n)))
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    dt = (time.perf_counter() - t0) / reps
    out.append(2 * m * k * n / dt / 1e9)
print(" ".join("%.1f" % g for g in out))
'''


def run(lib, threads, blas=None):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), KMP_DUPLICATE_LIB_OK="TRUE")
    if blas:
        env["PROMETHEUS_BLAS"] = blas
    r = subprocess.run([sys.executable, "-c", CHILD, lib, repr(SIZES)], env=env, capture_output=True, text=True)
    lines = (r.stdout or "").strip().splitlines()
    return lines[-1].split() if lines else ["-"] * len(SIZES)


print("GFLOP/s by m x k x n (higher is better)\n")
print("| backend | threads | " + " | ".join("%dx%dx%d" % s for s in SIZES) + " |")
print("|---|---|" + "---|" * len(SIZES))
for t in THREADS:
    for name, lib, blas in [("built-in", "prometheus", "tiled"), ("MKL", "prometheus", "mkl"), ("PyTorch", "torch", None)]:
        print("| %s | %d | %s |" % (name, t, " | ".join(run(lib, t, blas))), flush=True)
