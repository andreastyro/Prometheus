"""CPU matmul against float64 references: the forward product and both
backward products, through the built-in packed kernel (PROMETHEUS_BLAS=tiled)
and the BLAS library when one loads, at several OpenMP thread counts. The
shapes are large enough to take the parallel path, non-square, and not
multiples of the kernel's 6 x 16 register block or its 256-deep panels.

    python tests/test_gemm.py            # runs every (backend, threads) in a subprocess
"""
import os
import subprocess
import sys

import numpy as np

SHAPES = [(1, 1, 1), (5, 3, 7), (37, 53, 29), (130, 257, 1031), (517, 1023, 259),
          (3000, 7, 5), (7, 3000, 5), (5, 7, 3000), (6, 300, 16), (13, 513, 33)]


def run_one():
    import prometheus as p
    rng = np.random.default_rng(0)
    worst = 0.0
    for (m, k, n) in SHAPES:
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        g = rng.standard_normal((m, n)).astype(np.float32)
        A = p.Tensor([m, k], a.ravel().tolist())
        B = p.Tensor([k, n], b.ravel().tolist())
        A.requires_grad = True
        B.requires_grad = True
        y = p.matmul(A, B)
        G = p.Tensor([m, n], g.ravel().tolist())
        p.sum(p.reshape(y * G, [-1]), -1).backward()
        a64, b64, g64 = a.astype(np.float64), b.astype(np.float64), g.astype(np.float64)
        for got, want in [(y.numpy(), a64 @ b64),
                          (np.array(A.grad).reshape(m, k), g64 @ b64.T),
                          (np.array(B.grad).reshape(k, n), a64.T @ g64)]:
            # float32 sums over the inner dimension: error grows like sqrt(inner)
            scale = np.abs(want).max() + 1e-30
            err = float(np.abs(np.asarray(got, np.float64) - want).max() / scale)
            worst = max(worst, err)
            if err > 1e-5 * max(1.0, np.sqrt(max(m, k, n)) / 8):
                print("FAIL shape %s: relative error %.2e" % ((m, k, n), err))
                sys.exit(1)
    print("ok, worst relative error %.2e" % worst)


def main():
    if os.environ.get("TEST_GEMM_CHILD"):
        run_one()
        return
    failures = 0
    for backend in ("tiled", "auto"):
        for threads in (1, 2, 4, 8):
            env = dict(os.environ, TEST_GEMM_CHILD="1", PROMETHEUS_BLAS=backend, OMP_NUM_THREADS=str(threads),
                       KMP_DUPLICATE_LIB_OK="TRUE")
            out = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True)
            text = (out.stdout + out.stderr).strip().splitlines()
            line = text[-1] if text else "no output"
            ok = out.returncode == 0
            failures += 0 if ok else 1
            print("  %-6s %d threads: %s" % (backend, threads, line))
    print("gemm: %s" % ("PASS" if not failures else "%d FAIL" % failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
