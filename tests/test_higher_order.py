"""Higher-order gradients: p.grad with create_graph, checked against
independent references rather than against itself.

  - first-order p.grad agrees with .backward()
  - an analytic second derivative
  - Hessian-vector products against central finite differences of the
    gradient, on three models that between them cover every op with a
    differentiable backward
  - MAML's meta-gradient against finite differences of the meta-loss, and
    proof that its second-order term is not zero (so create_graph matters)
"""
import sys

import numpy as np
import prometheus as p

CHECKS, FAILURES = [0], []


def check(name, ok, detail=""):
    CHECKS[0] += 1
    if not ok:
        FAILURES.append(name)
    print("  %-58s %s %s" % (name, "ok" if ok else "** FAIL **", detail))


def T(a, grad=True):
    t = p.Tensor(list(a.shape), [float(v) for v in np.asarray(a).reshape(-1)])
    if grad:
        t.requires_grad = True
    return t


def A(t):
    return np.array(t.data).reshape(t.shape)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


rng = np.random.default_rng(0)


# ── models: f(params) -> scalar loss, built fresh from numpy each call ───────

X = rng.uniform(-1, 1, (6, 3))
Y = rng.uniform(-1, 1, (6, 2))
TARGETS = [int(v) for v in rng.integers(0, 4, 6)]


def mlp_tanh_mse(ps):
    W1, b1, W2, b2 = ps
    h = p.tanh_op(p.linear_op(T(X, False), W1, b1))
    return p.mse_loss(p.linear_op(h, W2, b2), T(Y, False))


def gelu_softmax_ce(ps):
    W1, b1, W2 = ps
    h = p.gelu(p.add(p.matmul(T(X, False), W1), b1))         # broadcast bias
    return p.cross_entropy_sparse_seq(p.matmul(h, W2), TARGETS)


XB = rng.uniform(-1, 1, (2, 3, 4))


def mixed_ops(ps):
    W, s = ps                                                  # [2, 4, 3], [4]
    z = p.divide(p.multiply(T(XB, False), s), p.add(1.5, p.sigmoid(T(XB, False))))
    m = p.matmul(z, W)                                         # batched [2,3,3]
    m = p.permute(p.reshape(m, [2, 9]), [1, 0])                # views
    e = p.sqrt_op(p.add(2.0, p.exp_op(p.multiply(0.3, m))))
    q = p.sum(p.log_op(p.add(3.0, p.softmax(e))), 0)           # reduce a dim
    return p.mean(p.multiply(q, q), -1)


MODELS = [
    ("MLP: linear + tanh + mse", mlp_tanh_mse,
     [(3, 5), (5,), (5, 2), (2,)]),
    ("gelu + broadcast add + matmul + softmax CE", gelu_softmax_ce,
     [(3, 5), (5,), (5, 4)]),
    ("broadcast mul/div, sigmoid, exp, sqrt, log, softmax, batched matmul, "
     "reshape, permute, sum(dim)", mixed_ops, [(2, 4, 3), (4,)]),
]


def grads_numpy(f, arrays):
    ps = [T(a) for a in arrays]
    return [A(g) for g in p.grad(f(ps), ps)]


def test_first_order():
    print("\n=== first order: p.grad agrees with .backward() ===\n")
    for name, f, shapes in MODELS:
        arrays = [rng.uniform(-0.8, 0.8, s) for s in shapes]
        ps = [T(a) for a in arrays]
        for t in ps:
            t.reset_grad()
        f(ps).backward()
        via_backward = [np.array(t.grad).reshape(t.shape) for t in ps]
        via_grad = grads_numpy(f, arrays)
        worst = max(rel(a, b) for a, b in zip(via_grad, via_backward))
        check(name[:58], worst < 1e-5, "rel err %.1e" % worst)


def test_second_order():
    print("\n=== second order ===\n")
    x0 = rng.uniform(-1, 1, (5,))
    x = T(x0)
    g, = p.grad(p.sum(p.pow_op(x, 3.0), -1), [x], create_graph=True)
    check("d/dx sum(x^3) == 3x^2", rel(A(g), 3 * x0 ** 2) < 1e-6)
    check("the gradient is itself in the graph", g.requires_grad)
    h, = p.grad(p.sum(g, -1), [x])
    check("d2/dx2 sum(x^3) == 6x", rel(A(h), 6 * x0) < 1e-5, "rel err %.1e" % rel(A(h), 6 * x0))

    # Hessian-vector product against finite differences of the gradient
    for name, f, shapes in MODELS:
        arrays = [rng.uniform(-0.8, 0.8, s) for s in shapes]
        vs = [rng.uniform(-1, 1, s) for s in shapes]
        ps = [T(a) for a in arrays]
        gs = p.grad(f(ps), ps, create_graph=True)
        dot = None
        for gi, vi in zip(gs, vs):
            term = p.sum(p.multiply(gi, T(vi, False)), -1)
            dot = term if dot is None else p.add(dot, term)
        hv = np.concatenate([A(t).reshape(-1) for t in p.grad(dot, ps)])

        eps = 1e-2
        plus = grads_numpy(f, [a + eps * v for a, v in zip(arrays, vs)])
        minus = grads_numpy(f, [a - eps * v for a, v in zip(arrays, vs)])
        fd = np.concatenate([((u - w) / (2 * eps)).reshape(-1) for u, w in zip(plus, minus)])
        e = rel(hv, fd)
        check("HVP == finite differences: " + name[:30], e < 2e-2, "rel err %.1e" % e)


def test_semantics():
    print("\n=== semantics ===\n")
    x = T(rng.uniform(-1, 1, (4,)))
    g, = p.grad(p.sum(p.multiply(x, x), -1), [x])
    check("without create_graph the gradient is a plain value", not g.requires_grad)

    x.reset_grad()
    p.grad(p.sum(p.multiply(x, x), -1), [x], create_graph=True)
    check("p.grad leaves .grad buffers alone", np.allclose(x.grad, 0.0))

    y = T(rng.uniform(-1, 1, (4,)))
    gx, gy = p.grad(p.sum(p.multiply(x, x), -1), [x, y])
    check("an unused input gets zeros", np.allclose(A(gy), 0.0))

    try:
        p.grad(p.sum(p.clip(x, -0.5, 0.5), -1), [x])
        check("an op without a differentiable backward raises", False)
    except Exception as err:
        check("an op without a differentiable backward raises, naming it",
              "clip" in str(err), str(err)[:50])


# ── MAML ──────────────────────────────────────────────────────────────────────

XS, YS = rng.uniform(-1, 1, (5, 1)), rng.uniform(-1, 1, (5, 1))   # support
XQ, YQ = rng.uniform(-1, 1, (5, 1)), rng.uniform(-1, 1, (5, 1))   # query
ALPHA = 0.1
SHAPES = [(1, 6), (6,), (6, 1), (1,)]


def net(ps, x):
    W1, b1, W2, b2 = ps
    return p.linear_op(p.tanh_op(p.linear_op(T(x, False), W1, b1)), W2, b2)


def meta_loss(ps, second_order):
    """Loss on the query set after one step on the support set."""
    inner = p.mse_loss(net(ps, XS), T(YS, False))
    gs = p.grad(inner, ps, create_graph=second_order)
    adapted = [p.subtract(w, p.multiply(ALPHA, g)) for w, g in zip(ps, gs)]
    return p.mse_loss(net(adapted, XQ), T(YQ, False))


def test_maml():
    print("\n=== MAML meta-gradient ===\n")
    arrays = [rng.uniform(-0.8, 0.8, s) for s in SHAPES]
    ps = [T(a) for a in arrays]
    meta = np.concatenate([A(g).reshape(-1) for g in p.grad(meta_loss(ps, True), ps)])
    ps = [T(a) for a in arrays]
    first = np.concatenate([A(g).reshape(-1) for g in p.grad(meta_loss(ps, False), ps)])

    def value(arrs):
        return meta_loss([T(a) for a in arrs], False).data[0]   # value needs no graph

    eps = 1e-2
    hits = []
    for _ in range(4):
        vs = [rng.uniform(-1, 1, s) for s in SHAPES]
        v = np.concatenate([x.reshape(-1) for x in vs])
        fd = (value([a + eps * x for a, x in zip(arrays, vs)]) -
              value([a - eps * x for a, x in zip(arrays, vs)])) / (2 * eps)
        hits.append((float(meta @ v), fd, float(first @ v)))
    e = max(abs(m - f) / max(abs(f), 1e-6) for m, f, _ in hits)
    check("meta-gradient == finite differences of the meta-loss", e < 2e-2, "rel err %.1e" % e)
    gap = rel(first, meta)
    check("the second-order term is not zero (create_graph matters)", gap > 1e-3,
          "first-order differs by %.1e" % gap)


def main():
    print("Higher-order gradients")
    test_first_order()
    test_second_order()
    test_semantics()
    test_maml()
    print("\n%d checks, %d failed" % (CHECKS[0], len(FAILURES)))
    print("PASS" if not FAILURES else "FAIL")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
