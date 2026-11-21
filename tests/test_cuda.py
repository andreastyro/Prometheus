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


def test_compile():
    print("\n=== prometheus.compile: a captured step replays the eager one ===\n")
    data = p.Tensor([2, 12], [float(i % 30) for i in range(24)], p.int64).cuda()
    tg = p.Tensor([24], [float((i * 7 + 3) % 30) for i in range(24)], p.int64).cuda()
    curves = []
    for compiled in (False, True):
        p.Tensor.manual_seed(11)
        model = p.GPT(30, 16, 32, 4, 2)
        model.cuda()
        opt = p.AdamW(model.parameters(), 3e-3)

        def step(x, t):
            opt.zero_grad()
            loss = p.cross_entropy_sparse_seq(model.forward(x), t)
            loss.backward()
            opt.step()
            return loss

        run = p.compile(step, warmup=2) if compiled else step
        curves.append([run(data, tg).item() for _ in range(8)])
        if compiled:
            check("compile: the step was captured", run.graph is not None and not run.eager_only)
    e, c = np.array(curves[0]), np.array(curves[1])
    check("compile: replayed losses match the eager run", np.abs(e - c).max() / e[0] < 1e-4)

    def reads_back(x):
        return p.sum(x, -1).item()
    f = p.compile(reads_back, warmup=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = f(p.Tensor([4], [1.0, 2.0, 3.0, 4.0]).cuda())
    check("compile: a step that reads a value back runs eagerly", f.eager_only and abs(v - 10.0) < 1e-6)


def test_prefetch():
    print("\n=== prefetch: batches upload on a second stream ===\n")
    rng = np.random.default_rng(3)
    host = [(T(rng.standard_normal((64, 33)).astype(np.float32)), T(rng.standard_normal(64).astype(np.float32)))
            for _ in range(6)]
    sums = []
    for x, y in p.prefetch(host, depth=3):
        # compute on each batch right away: the copy must have landed first
        sums.append(p.sum(p.reshape(x * 2.0, [-1]), -1).item() + p.sum(y, -1).item())
    want = [2.0 * float(A(x).sum()) + float(A(y).sum()) for x, y in host]
    check("prefetch: every batch arrives on the GPU intact, in order",
          len(sums) == 6 and max(abs(a - b) for a, b in zip(sums, want)) < 1e-2)
    check("prefetch: GPU tensors out", all(t.device == p.cuda for t in next(iter(p.prefetch([host[0]])))))


def test_fuse():
    print("\n=== fuse: an element-wise function as one kernel ===\n")

    def f(x, t):
        y = x * p.fx.sigmoid(x)
        return (y - t) * (y - t) + p.fx.gelu(x) / 3.0 - p.fx.maximum(x, t) ** 2 + p.fx.exp(-p.fx.abs(t))
    fused = p.fuse(f)
    rng = np.random.default_rng(7)
    xv = rng.standard_normal((5, 7)).astype(np.float32)
    tv = rng.standard_normal(7).astype(np.float32)          # broadcast along the rows
    out = {}
    for dev in ("cpu", "cuda"):
        x, t = T(xv), T(tv)
        if dev == "cuda":
            x, t = x.cuda(), t.cuda()
        x.requires_grad = True
        t.requires_grad = True
        y = (fused if dev == "cuda" else f)(x, t)
        p.sum(p.reshape(y, [-1]), -1).backward()
        out[dev] = (A(y.cpu() if dev == "cuda" else y), np.array(x.grad).reshape(xv.shape),
                    np.array(t.grad).reshape(tv.shape))
    check("fuse: GPU kernel matches the ops on the CPU", np.abs(out["cuda"][0] - out["cpu"][0]).max() < 1e-4)
    check("fuse: gradient of each input (one broadcast)",
          np.abs(out["cuda"][1] - out["cpu"][1]).max() < 1e-3 and np.abs(out["cuda"][2] - out["cpu"][2]).max() < 1e-3)
    # a single same-shape op: library or generated kernel, chosen by timing both once
    from prometheus.compiler import executor
    add = p.fuse(lambda a, b: a * b)
    r = add(T(xv).cuda(), T(xv).cuda())
    check("fuse: single op measured, the faster kept, the result right",
          ("mul", tuple(xv.shape)) in executor.tuned() and np.abs(r.cpu().numpy() - xv * xv).max() < 1e-5)

    # half inputs stay half: read and written in their dtype, computed in float32
    xh, th = T(xv).cuda().to(p.float16), T(tv).cuda().to(p.float16)
    xh.requires_grad = True
    yh = fused(xh, th)
    p.sum(p.reshape(yh, [-1]), -1).backward()
    y32 = fused(T(xv).cuda(), T(tv).cuda())
    check("fuse: float16 in, float16 out", yh.dtype == p.float16)
    check("fuse: float16 result within half precision of float32",
          np.abs(yh.to(p.float32).cpu().numpy() - y32.cpu().numpy()).max() / np.abs(y32.cpu().numpy()).max() < 5e-3)
    check("fuse: float16 input gets a gradient", np.isfinite(np.array(xh.grad)).all())

    # a reduction inside: two generated kernels around the library's mean
    def loss(x, t):
        y = x * p.fx.sigmoid(x) - t
        return p.fx.mean(y * y) + p.fx.sum(p.fx.relu(x), 0) / 5.0
    lf = p.fuse(loss)
    lo = {}
    for dev in ("cpu", "cuda"):
        x, t = T(xv), T(tv)
        if dev == "cuda":
            x, t = x.cuda(), t.cuda()
        x.requires_grad = True
        y = (lf if dev == "cuda" else loss)(x, t)
        p.sum(p.reshape(y, [-1]), -1).backward()
        lo[dev] = ((y.cpu() if dev == "cuda" else y).numpy(), np.array(x.grad).reshape(xv.shape))
    check("fuse: reductions inside (mean, sum along an axis)", np.abs(lo["cuda"][0] - lo["cpu"][0]).max() < 1e-4)
    check("fuse: gradient through the reductions", np.abs(lo["cuda"][1] - lo["cpu"][1]).max() < 1e-4)

    # a whole block traced: LayerNorm, two matmuls, GELU, residual -- library
    # ops and generated kernels; every parameter's gradient against the CPU
    def block(x, g, b, w1, b1, w2, b2):
        h = p.fx.layernorm(x, g, b)
        h = p.fx.gelu(h @ w1 + b1)
        return h @ w2 + b2 + x
    bf = p.fuse(block)
    shapes = [(6, 16), (16,), (16,), (16, 24), (24,), (24, 16), (16,)]
    vals = [rng.standard_normal(s).astype(np.float32) * 0.5 for s in shapes]
    res = {}
    for dev in ("cpu", "cuda"):
        ts = [T(v) for v in vals]
        if dev == "cuda":
            ts = [t.cuda() for t in ts]
        for t in ts:
            t.requires_grad = True
        y = (bf if dev == "cuda" else block)(*ts)
        p.sum(p.reshape(y * y, [-1]), -1).backward()
        res[dev] = [(y.cpu() if dev == "cuda" else y).numpy()] + [np.array(t.grad).reshape(-1) for t in ts]
    check("fuse: a whole block (LayerNorm, matmuls, GELU, residual)", np.abs(res["cuda"][0] - res["cpu"][0]).max() < 1e-4)
    check("fuse: every parameter's gradient through the block",
          max(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6) for a, b in zip(res["cuda"][1:], res["cpu"][1:])) < 1e-4)

    # second derivative through a fused function: its gradient is fused again
    swish = p.fuse(lambda x: x * p.fx.sigmoid(x) + p.fx.tanh(x) * x)
    hs = {}
    for dev in ("cpu", "cuda"):
        w = T([0.7, -1.3, 0.2])
        if dev == "cuda":
            w = w.cuda()
        w.requires_grad = True
        fn = swish if dev == "cuda" else swish.fn
        (g,) = p.grad(p.sum(fn(w), -1), [w], create_graph=True)
        (h,) = p.grad(p.sum(g, -1), [w])
        hs[dev] = (h.cpu() if dev == "cuda" else h).numpy().reshape(-1)
    check("fuse: second derivative (the gradient fused again)", np.abs(hs["cuda"] - hs["cpu"]).max() < 1e-4)

    p.cuda_reset_launch_counts()
    fused(T(xv).cuda(), T(tv).cuda())
    p.cuda_synchronize()
    check("fuse: one kernel forward", p.cuda_launch_counts()["kernels"] <= 1)


def main():
    print("GPU (Python)")
    print("  " + p.cuda_status())
    if not p.cuda_available():
        print("no GPU: skipped")
        return
    test_basics()
    test_gradients()
    test_training()
    test_compile()
    test_prefetch()
    test_fuse()
    print("\n%d checks, %d failed" % (CHECKS[0], len(FAILURES)))
    print("PASS" if not FAILURES else "FAIL")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
