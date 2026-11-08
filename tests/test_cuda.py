"""The GPU from Python: the same results as the CPU, and a model that trains.

Skips (passes) on a machine without an NVIDIA GPU. NVRTC and cuBLAS come from
pip (nvidia-cuda-nvrtc-cu12, nvidia-cublas-cu12), which the module finds on
its own -- that is part of what this checks.
"""
import sys

import numpy as np
import prometheus as p

CHECKS, FAILURES = [0], []


def check(name, ok, detail=""):
    CHECKS[0] += 1
    if not ok:
        FAILURES.append(name)
    print("  %-60s %s %s" % (name, "ok" if ok else "** FAIL **", detail))


def T(a, dtype=p.float32):
    a = np.asarray(a)
    return p.Tensor(list(a.shape), [float(v) for v in a.reshape(-1)], dtype)


def A(t):
    return np.array(t.numpy(), dtype=np.float64)


def rel(gpu, cpu):
    a, b = A(gpu), A(cpu)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def test_basics():
    print("\n=== tensors and ops ===\n")
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2, (4, 6))
    xc, xg = T(x), T(x).cuda()
    check("t.cuda() is on the GPU", xg.device == p.cuda)
    check("t.cpu() round trip is exact", np.array_equal(A(xg.cpu()), A(xc)))
    check("numpy() of a GPU tensor", np.array_equal(xg.numpy(), xc.numpy()))
    check("operators on the GPU", rel(xg * 2 + xg / 3 - 1, xc * 2 + xc / 3 - 1) < 1e-6)
    check("matmul (cuBLAS)", rel(xg @ p.transpose(xg), xc @ p.transpose(xc)) < 1e-6)
    check("softmax", rel(p.softmax(xg), p.softmax(xc)) < 1e-6)
    check("comparisons and masks", np.array_equal(A(p.masked_fill(xg, xg > 1.0, 0.0)),
                                                  A(p.masked_fill(xc, xc > 1.0, 0.0))))
    check("item() from the GPU", abs(p.sum(xg, -1).item() - p.sum(xc, -1).item()) < 1e-4)
    check("mixing devices raises", raises(lambda: xg + T(x)))


def raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def test_gradients():
    print("\n=== gradients on the GPU ===\n")
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 7))
    ac, ag = T(a), T(a).cuda()
    ac.requires_grad = True
    ag.requires_grad = True
    for t in (ac, ag):
        p.sum(p.gelu(t) * p.softmax(t), -1).backward()
    check("backward on the GPU matches the CPU",
          np.abs(np.array(ag.grad) - np.array(ac.grad)).max() < 1e-6)
    w = T([0.7, -1.3]).cuda()
    w.requires_grad = True
    (g,) = p.grad(p.sum(p.tanh_op(w), -1), [w], create_graph=True)
    (h,) = p.grad(p.sum(g, -1), [w])
    tw = np.tanh([0.7, -1.3])
    check("second derivative on the GPU", np.abs(A(h) + 2 * tw * (1 - tw ** 2)).max() < 1e-6)


def test_training():
    print("\n=== a GPT trains on the GPU, tracking the CPU ===\n")
    ids = list(range(1, 25))
    data = p.Tensor([2, 12], [float(i % 30) for i in ids], p.int64)
    targets = [(i * 7 + 3) % 30 for i in range(24)]
    curves = {}
    for dev in ("cpu", "cuda"):
        p.Tensor.manual_seed(11)
        model = p.GPT(30, 16, 32, 4, 2)
        opt = p.AdamW(model.parameters(), 3e-3)
        x = data
        if dev == "cuda":
            model.cuda()
            x = data.cuda()
        losses = []
        for _ in range(10):
            opt.zero_grad()
            loss = p.cross_entropy_sparse_seq(model.forward(x), targets)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        curves[dev] = losses
        if dev == "cuda":
            check("model.cuda() moved every parameter",
                  all(t.device == p.cuda for t in model.parameters()))
    c, g = np.array(curves["cpu"]), np.array(curves["cuda"])
    print("  loss %.4f -> cpu %.4f, gpu %.4f" % (c[0], c[-1], g[-1]))
    check("step 0: the same loss", abs(c[0] - g[0]) / c[0] < 1e-5)
    check("the GPU run learns", g[-1] < 0.8 * g[0])
    check("the curves agree within float drift (10 AdamW steps)", np.abs(c - g).max() / c[0] < 1e-3)


def main():
    print("GPU (Python)")
    print("  " + p.cuda_status())
    if not p.cuda_available():
        print("no GPU: skipped")
        return
    test_basics()
    test_gradients()
    test_training()
    print("\n%d checks, %d failed" % (CHECKS[0], len(FAILURES)))
    print("PASS" if not FAILURES else "FAIL")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
