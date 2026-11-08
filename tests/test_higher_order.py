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

    # every public op now has a differentiable backward; clip's is checked
    # against an analytic second derivative of a function routed through it
    xc = T(np.array([0.1, -0.3, 0.45, 0.8, -0.9]))
    gc, = p.grad(p.sum(p.pow_op(p.clip(xc, -0.5, 0.5), 3.0), -1), [xc], create_graph=True)
    hc, = p.grad(p.sum(gc, -1), [xc])
    xv = np.array([0.1, -0.3, 0.45, 0.8, -0.9])
    inside = (xv > -0.5) & (xv < 0.5)
    check("clip: second derivative through it", np.allclose(A(hc), np.where(inside, 6 * xv, 0.0), atol=1e-5))


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


# ── every layer ──────────────────────────────────────────────────────────────
#
# Each fast kernel's higher-order backward differentiates a reference forward
# built from differentiable ops. Two independent checks per layer:
#   1. p.grad -- which runs the reference -- equals .backward(), which runs the
#      hand-written kernel. So the reference computes the same function.
#   2. Hessian-vector products equal finite differences of .backward()'s
#      gradient, so the second derivative is right.
# The loss is mean(tanh(out) * w): tanh gives every layer, linear ones
# included, a non-zero second derivative to check.

def layer_check(name, forward, tensors, fd=True, eps=1e-2, tol=2e-2):
    W = [None]
    def loss(ts):
        out = forward(ts)
        if W[0] is None:
            W[0] = T(np.random.default_rng(7).uniform(-1, 1, out.shape), grad=False)
        return p.mean(p.multiply(p.tanh_op(out), W[0]), -1)

    # 1. reference (p.grad) against the fused kernel (.backward)
    for t in tensors:
        t.reset_grad()
    l = loss(tensors)
    via_grad = [A(g) for g in p.grad(l, tensors)]
    l.backward()
    via_kernel = [np.array(t.grad).reshape(t.shape) for t in tensors]
    e1 = max(rel(a, b) for a, b in zip(via_grad, via_kernel))
    check("%-24s reference == kernel" % name, e1 < 1e-4, "rel err %.1e" % e1)
    if not fd:
        return

    # 2. Hessian-vector product against finite differences of the gradient
    base = [A(t) for t in tensors]
    vs = [np.random.default_rng(8 + i).uniform(-1, 1, b.shape) for i, b in enumerate(base)]
    gs = p.grad(loss(tensors), tensors, create_graph=True)
    dot = None
    for g, v in zip(gs, vs):
        term = p.sum(p.multiply(g, T(v, grad=False)), -1)
        dot = term if dot is None else p.add(dot, term)
    hv = np.concatenate([A(h).reshape(-1) for h in p.grad(dot, tensors)])

    def grads_at(step):
        for t, b, v in zip(tensors, base, vs):
            t.data = list((b + step * v).reshape(-1))
            t.reset_grad()
        loss(tensors).backward()
        return np.concatenate([np.array(t.grad) for t in tensors])
    def central(h):
        return (grads_at(+h) - grads_at(-h)) / (2 * h)
    # Richardson extrapolation: a central difference is off by c*h^2; steps h
    # and h/2 together cancel it. Without it, attention through a sharp
    # softmax left several percent of truncation error at h = 1e-2 -- error
    # that fell as h^2, the signature of a correct HVP measured too coarsely.
    fd_hv = (4 * central(eps / 2) - central(eps)) / 3
    for t, b in zip(tensors, base):
        t.data = list(b.reshape(-1))
    e2 = rel(hv, fd_hv)
    check("%-24s HVP == finite diff" % name, e2 < tol, "rel err %.1e" % e2)


def test_layers():
    print("\n=== every layer: reference == kernel, HVP == finite differences ===\n")
    rng = np.random.default_rng(1)
    def X(*shape, lo=-1.0, hi=1.0):
        return T(rng.uniform(lo, hi, shape))
    p.Tensor.manual_seed(3)

    # core ops that were missing one
    a, b = X(3, 4), X(5, 4)
    layer_check("abs_op", lambda ts: p.abs_op(ts[0]), [X(3, 4, lo=0.2) if False else a])
    layer_check("max_op / min_op", lambda ts: p.add(p.max_op(ts[0]), p.min_op(ts[0])), [X(12)])
    c1, c2 = X(2, 3), X(4, 3)
    layer_check("concat_op", lambda ts: p.concat_op([ts[0], ts[1]], 0), [c1, c2])
    layer_check("matmul_bt", lambda ts: p.matmul_bt(ts[0], ts[1]), [a, b])

    conv = p.Conv2D(3, 4, 3, 1, 1)
    layer_check("Conv2D", lambda ts: conv.forward(ts[0]), [X(2, 3, 6, 6)] + conv.parameters())
    conv2 = p.Conv2D(3, 4, 3, 2, 1)
    layer_check("Conv2D stride 2", lambda ts: conv2.forward(ts[0]), [X(2, 3, 7, 7)] + conv2.parameters())
    ct = p.ConvTranspose2D(4, 3, 4, 2, 1)
    layer_check("ConvTranspose2D", lambda ts: ct.forward(ts[0]), [X(2, 4, 3, 3)] + ct.parameters())
    mp, ap = p.MaxPool2D(2), p.AvgPool2D(2)
    layer_check("MaxPool2D", lambda ts: mp.forward(ts[0]), [X(2, 3, 6, 6)])
    layer_check("AvgPool2D", lambda ts: ap.forward(ts[0]), [X(2, 3, 6, 6)])

    bn = p.BatchNorm(3)
    layer_check("BatchNorm (train)", lambda ts: bn.forward(ts[0]), [X(4, 3, 3, 3)] + bn.parameters())
    bn_e = p.BatchNorm(3)
    bn_e.forward(X(4, 3, 3, 3))
    bn_e.eval()
    layer_check("BatchNorm (eval)", lambda ts: bn_e.forward(ts[0]), [X(4, 3, 3, 3)] + bn_e.parameters())
    gn = p.GroupNorm(2, 4)
    layer_check("GroupNorm", lambda ts: gn.forward(ts[0]), [X(2, 4, 3, 3)] + gn.parameters())
    ln = p.LayerNorm(6)
    layer_check("LayerNorm", lambda ts: ln.forward(ts[0]), [X(3, 2, 6)] + ln.parameters())

    emb = p.Embedding(10, 4)
    ids = p.Tensor([2, 5], [float(v) for v in [1, 3, 3, 7, 0, 9, 1, 2, 3, 3]], p.int64)
    layer_check("Embedding", lambda ts: emb.forward(ids), emb.parameters())

    for causal in (False, True):
        for rope in (False, True):
            mha = p.MultiHeadAttention(8, 2, causal, rope)
            layer_check("attention c=%d r=%d" % (causal, rope),
                        lambda ts, m=mha: m.forward(ts[0]), [X(2, 5, 8)] + mha.parameters())

    for name, layer in (("RNN", p.RNN(5, 6)), ("LSTM", p.LSTM(5, 6)), ("GRU", p.GRU(5, 6))):
        layer_check(name, lambda ts, L=layer: L.forward(ts[0]), [X(4, 3, 5)] + layer.parameters())

    fl, pe = p.Flatten(), p.PositionalEncoding(10, 6)
    layer_check("Flatten", lambda ts: fl.forward(ts[0]), [X(2, 3, 4)])
    layer_check("PositionalEncoding", lambda ts: pe.forward(ts[0]), [X(5, 6)])
    dr = p.Dropout(0.3)
    # a new mask every forward pass, so finite differences do not apply
    layer_check("Dropout", lambda ts: dr.forward(ts[0]), [X(4, 6)], fd=False)

    gpt = p.GPT(20, 6, 8, 2, 1)
    gids = p.Tensor([2, 6], [float(v) for v in rng.integers(0, 20, 12)], p.int64)
    tg = [int(v) for v in rng.integers(0, 20, 12)]
    # the loss is scaled down because the harness takes its tanh: at a
    # cross entropy of ~3 tanh saturates, and its curvature would be lost
    layer_check("GPT (whole model)",
                lambda ts: p.multiply(0.1, p.reshape(p.cross_entropy_sparse_seq(gpt.forward(gids), tg), [1])),
                gpt.parameters())

    # the losses that were missing one
    pr, tgt = X(3, 4), T(rng.uniform(-1, 1, (3, 4)), grad=False)
    layer_check("huber_loss", lambda ts: p.reshape(p.huber_loss(ts[0], tgt, 0.5), [1]), [pr])
    an, po, ne = X(6), X(6), X(6)
    layer_check("contrastive_loss", lambda ts: p.contrastive_loss(ts[0], ts[1], ts[2], 3.0), [an, po, ne])
    w1, w2 = X(3, 3), X(4)
    layer_check("l1_regularization", lambda ts: p.l1_regularization([ts[0], ts[1]], 0.1), [w1, w2])
    layer_check("l2_regularization", lambda ts: p.l2_regularization([ts[0], ts[1]], 0.1), [w1, w2])
    lg = X(7)
    layer_check("cross_entropy_sparse", lambda ts: p.cross_entropy_sparse(ts[0], 3), [lg])


def main():
    print("Higher-order gradients")
    test_first_order()
    test_second_order()
    test_semantics()
    test_maml()
    test_layers()
    print("\n%d checks, %d failed" % (CHECKS[0], len(FAILURES)))
    print("PASS" if not FAILURES else "FAIL")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
