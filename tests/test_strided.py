"""Non-contiguous inputs: every op here run on a strided view (a transpose,
a column slice) must give what it gives on a contiguous copy of the same
values -- the result and the input's gradient. On the CPU, and on the GPU
when there is one.

    python tests/test_strided.py
"""
import sys

import numpy as np
import prometheus as p

CHECKS = [0]
FAILURES = []


def check(name, ok):
    CHECKS[0] += 1
    print("  %-62s %s" % (name, "ok" if ok else "** FAIL **"))
    if not ok:
        FAILURES.append(name)


def views(dev):
    """(name, a function making the strided view, its values as numpy)."""
    rng = np.random.default_rng(3)
    put = (lambda t: t.cuda()) if dev == "cuda" else (lambda t: t)
    a = rng.standard_normal((6, 10)).astype(np.float32)
    base = put(p.Tensor([10, 6], a.T.copy().ravel().tolist()))          # transposed: strides (1, 6)
    c = rng.standard_normal((3, 4, 5)).astype(np.float32)
    cube = put(p.Tensor([3, 4, 5], c.ravel().tolist()))                  # permuted to [5, 3, 4]: no unit stride last
    return [("transpose", lambda: p.transpose(base), a),
            ("3-D permutation", lambda: p.permute(cube, [2, 0, 1]), np.transpose(c, (2, 0, 1)).copy())]


OPS = [
    ("add", lambda x, y: x + y, True),
    ("multiply", lambda x, y: x * y, True),
    ("exp", lambda x, y: p.exp_op(x), False),
    ("gelu", lambda x, y: p.gelu(x), False),
    ("sigmoid", lambda x, y: p.sigmoid(x), False),
    ("softmax", lambda x, y: p.softmax(x), False),
    ("sum over the last dim", lambda x, y: p.sum(x, len(x.shape) - 1, True), False),
    ("matmul", lambda x, y: p.matmul(x, p.transpose(y)), True),
    ("cast to float16 and back", lambda x, y: p.cast(p.cast(x, p.float16), p.float32), False),
]


def run(dev):
    print("\n=== strided inputs, %s ===\n" % dev)
    rng = np.random.default_rng(5)
    put = (lambda t: t.cuda()) if dev == "cuda" else (lambda t: t)
    for vname, make, vals in views(dev):
        shape = list(vals.shape)
        y = put(p.Tensor(shape, rng.standard_normal(vals.shape).astype(np.float32).ravel().tolist()))
        for oname, op, _ in OPS:
            outs = []
            for x in (make(), put(p.Tensor(shape, vals.ravel().tolist()))):      # the view, then a copy
                try:
                    out = op(x, y)
                    outs.append(np.array((out.cpu() if dev == "cuda" else out).to(p.float32).numpy()))
                except Exception as e:  # noqa: BLE001
                    outs.append(e)
            ok = (all(isinstance(o, np.ndarray) for o in outs) and outs[0].shape == outs[1].shape
                  and np.allclose(outs[0], outs[1], rtol=1e-5, atol=1e-5))
            check("%s on a %s" % (oname, vname), ok)


def run_grads(dev):
    print("\n=== gradients through strided views, %s ===\n" % dev)
    rng = np.random.default_rng(7)
    a = rng.standard_normal((6, 10)).astype(np.float32)
    w = rng.standard_normal((10, 6)).astype(np.float32)
    grads = []
    for strided in (True, False):
        if strided:
            base = p.Tensor([10, 6], a.T.copy().ravel().tolist())
        else:
            base = p.Tensor([6, 10], a.ravel().tolist())
        if dev == "cuda":
            base = base.cuda()
        base.requires_grad = True
        x = p.transpose(base) if strided else base
        W = p.Tensor([10, 6], w.ravel().tolist())
        W = W.cuda() if dev == "cuda" else W
        y = p.gelu(p.matmul(x, W)) * p.sigmoid(p.matmul(x, W))
        p.sum(p.reshape(p.softmax(y), [-1]), -1).backward()
        g = np.array(base.grad)
        grads.append(g.reshape(6, 10).T if strided else g.reshape(6, 10).T)
    check("gradient through a transposed input matches the contiguous one",
          np.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-6))


def main():
    print("strided tensors")
    devices = ["cpu"] + (["cuda"] if p.cuda_available() else [])
    for dev in devices:
        run(dev)
        run_grads(dev)
    print("\n%d checks, %d failed" % (CHECKS[0], len(FAILURES)))
    print("PASS" if not FAILURES else "FAIL")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
