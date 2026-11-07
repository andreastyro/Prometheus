"""
Forward-pass verification against independent numpy reference implementations.

gradcheck proves the backward pass agrees with the forward pass. It cannot
prove the forward pass is right: a wrong-but-self-consistent forward passes
gradcheck happily. This file closes that gap by recomputing every layer, op and
loss in numpy from the same weights and comparing the results.

The numpy side is written from the mathematical definition, not from the C++
source, so an error shared by both would have to be made twice independently.

Run directly:  python tests/test_forward_reference.py
"""

import sys

import numpy as np

import prometheus as p

FAILURES = []
CHECKS = [0]


def check(name, got, want, atol=1e-5, rtol=0.0):
    """Compare a prometheus result against the numpy reference.

    Passes where |got - want| <= atol + rtol * |want| everywhere, the same
    criterion as numpy.allclose. rtol defaults to 0, so a check is purely
    absolute unless it asks otherwise.
    """
    CHECKS[0] += 1
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)

    if got.shape != want.shape:
        FAILURES.append("%s: shape %s, reference %s" % (name, got.shape, want.shape))
        print("  %-46s shape %-14s expected %-14s ** FAIL **"
              % (name, got.shape, want.shape))
        return

    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    ok = bool(np.all(np.abs(got - want) <= atol + rtol * np.abs(want)))
    if not ok:
        FAILURES.append("%s: max abs error %.3e" % (name, err))
    print("  %-46s max abs err %.3e   %s" % (name, err, "ok" if ok else "** FAIL **"))


def arr(t):
    """prometheus Tensor -> numpy array with its shape."""
    return np.array(t.data, dtype=np.float64).reshape(t.shape)


def tensor(shape, values):
    t = p.Tensor(shape)
    t.data = [float(v) for v in np.asarray(values).ravel()]
    return t


def spread(n, lo=-0.9, hi=0.9):
    """Deterministic values across a range, avoiding exact zeros."""
    return np.linspace(lo, hi, n) + 0.013


# ---------------------------------------------------------------- element-wise
def test_ops():
    print("\n=== element-wise ops and reductions ===")
    a_np = spread(12).reshape(3, 4)
    b_np = (spread(12)[::-1] * 0.7).reshape(3, 4)
    a, b = tensor([3, 4], a_np), tensor([3, 4], b_np)

    check("add",      arr(p.add(a, b)),      a_np + b_np)
    check("subtract", arr(p.subtract(a, b)), a_np - b_np)
    check("multiply", arr(p.multiply(a, b)), a_np * b_np)
    check("divide",   arr(p.divide(a, b)),   a_np / b_np)

    check("relu",    arr(p.relu(a)),    np.maximum(a_np, 0))
    check("sigmoid", arr(p.sigmoid(a)), 1 / (1 + np.exp(-a_np)))
    check("tanh_op", arr(p.tanh_op(a)), np.tanh(a_np))
    check("exp_op",  arr(p.exp_op(a)),  np.exp(a_np))
    check("abs_op",  arr(p.abs_op(a)),  np.abs(a_np))

    pos = tensor([3, 4], np.abs(a_np) + 0.2)
    check("log_op",  arr(p.log_op(pos)),  np.log(np.abs(a_np) + 0.2))
    check("sqrt_op", arr(p.sqrt_op(pos)), np.sqrt(np.abs(a_np) + 0.2))

    check("pow_op(2)", arr(p.pow_op(a, 2.0)), a_np ** 2)
    check("clip",      arr(p.clip(a, -0.3, 0.4)), np.clip(a_np, -0.3, 0.4))

    # GELU, tanh approximation (the form used inside GPT-2)
    gelu_ref = 0.5 * a_np * (1 + np.tanh(np.sqrt(2 / np.pi) * (a_np + 0.044715 * a_np ** 3)))
    check("gelu", arr(p.gelu(a)), gelu_ref, atol=1e-4)

    # softmax, row-wise and numerically stabilised
    sm = np.exp(a_np - a_np.max(axis=-1, keepdims=True))
    sm = sm / sm.sum(axis=-1, keepdims=True)
    check("softmax (row-wise)", arr(p.softmax(a)), sm)

    # Large logits. This is the case the check above cannot see: with values in
    # [-0.9, 0.9] an unstabilised softmax agrees with a stabilised one to 1e-8,
    # which is exactly why the missing max-subtraction went unnoticed. exp(90)
    # already overflows float32.
    for mx in (50.0, 90.0, 200.0, 1e4):
        row = np.array([[mx, mx - 1.0, mx - 2.0, mx - 3.0]])
        ref = np.exp(row - row.max(axis=-1, keepdims=True))
        ref = ref / ref.sum(axis=-1, keepdims=True)
        got = arr(p.softmax(tensor([1, 4], row)))
        check("softmax with max logit %g" % mx, got, ref, atol=1e-6)
        check("softmax rows sum to 1 at max logit %g" % mx,
              got.sum(), 1.0, atol=1e-5)

    m_np = spread(8).reshape(4, 2)
    check("matmul", arr(p.matmul(a, tensor([4, 2], m_np))), a_np @ m_np)

    check("transpose_op", arr(p.transpose_op(a)), a_np.T)

    # a @ b^T computed with the transpose as a flag rather than a copy, which
    # is what nn.Linear does in PyTorch. Must agree with both numpy and the
    # explicit-transpose route.
    bt = spread(8).reshape(2, 4)
    check("matmul_bt", arr(p.matmul_bt(a, tensor([2, 4], bt))), a_np @ bt.T)
    check("matmul_bt == matmul(a, transpose_op(b))",
          arr(p.matmul_bt(a, tensor([2, 4], bt))),
          arr(p.matmul(a, p.transpose_op(tensor([2, 4], bt)))))
    check("reshape_op",   arr(p.reshape_op(a, [4, 3])), a_np.reshape(4, 3))

    check("max_op", arr(p.max_op(a)), np.array([a_np.max()]))
    check("min_op", arr(p.min_op(a)), np.array([a_np.min()]))


# ----------------------------------------------------------------------- dense
def test_linear():
    print("\n=== Linear ===")
    fc = p.Linear(3, 2)                     # weights [in, out], out = x @ W + b
    W, bias = arr(fc.weights), arr(fc.bias)
    x_np = spread(6).reshape(2, 3)
    check("Linear(3 -> 2)", arr(fc.forward(tensor([2, 3], x_np))), x_np @ W + bias)

    fc2 = p.Linear(5, 4)
    W2, b2 = arr(fc2.weights), arr(fc2.bias)
    x2 = spread(15).reshape(3, 5)
    check("Linear(5 -> 4), batch 3",
          arr(fc2.forward(tensor([3, 5], x2))), x2 @ W2 + b2)


def test_embedding_flatten():
    print("\n=== Embedding and Flatten ===")
    emb = p.Embedding(6, 4)                 # weight [vocab, dim]
    W = arr(emb.weight)
    ids = [1, 4, 1]
    out = arr(emb.forward(tensor([1, 3], ids)))
    check("Embedding lookup (repeated id)", out.reshape(3, 4), W[ids])

    fl = p.Flatten()
    x_np = spread(2 * 1 * 3 * 3).reshape(2, 1, 3, 3)
    check("Flatten", arr(fl.forward(tensor([2, 1, 3, 3], x_np))), x_np.reshape(2, 9))


# ------------------------------------------------------------------ normalisers
def test_norms():
    print("\n=== normalisation layers ===")
    ln = p.LayerNorm(4)
    g, b = arr(ln.parameters()[0]), arr(ln.parameters()[1])
    x_np = spread(12).reshape(3, 4)
    mu = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)
    ref = g * (x_np - mu) / np.sqrt(var + 1e-5) + b
    check("LayerNorm (per row)", arr(ln.forward(tensor([3, 4], x_np))), ref, atol=1e-4)

    # GroupNorm: statistics over (channels in group) x spatial, per batch item
    gn = p.GroupNorm(2, 4)
    gg, gb = arr(gn.parameters()[0]), arr(gn.parameters()[1])
    xg = spread(1 * 4 * 2 * 2).reshape(1, 4, 2, 2)
    ref = np.empty_like(xg)
    for bi in range(1):
        for grp in range(2):
            sl = xg[bi, grp * 2:(grp + 1) * 2]
            mean, var = sl.mean(), sl.var()
            ref[bi, grp * 2:(grp + 1) * 2] = (sl - mean) / np.sqrt(var + 1e-5)
    ref = ref * gg.reshape(1, 4, 1, 1) + gb.reshape(1, 4, 1, 1)
    check("GroupNorm (2 groups of 2)",
          arr(gn.forward(tensor([1, 4, 2, 2], xg))), ref, atol=1e-4)

    # BatchNorm: statistics down the batch axis, per feature
    bn = p.BatchNorm(4)
    bg, bb = arr(bn.parameters()[0]), arr(bn.parameters()[1])
    xb = spread(12).reshape(3, 4)
    mu = xb.mean(axis=0, keepdims=True)
    var = xb.var(axis=0, keepdims=True)
    ref = bg * (xb - mu) / np.sqrt(var + 1e-5) + bb
    check("BatchNorm (per feature)",
          arr(bn.forward(tensor([3, 4], xb))), ref, atol=1e-4)


# ---------------------------------------------------------------------- vision
def conv2d_ref(x, k, bias, stride, pad):
    """Direct convolution (cross-correlation), NCHW with OIHW kernels."""
    n, cin, h, w = x.shape
    cout, _, kh, kw = k.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    out = np.zeros((n, cout, oh, ow))
    for ni in range(n):
        for co in range(cout):
            for i in range(oh):
                for j in range(ow):
                    patch = xp[ni, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
                    out[ni, co, i, j] = np.sum(patch * k[co]) + bias[co]
    return out


def test_vision():
    print("\n=== vision layers ===")
    for stride, pad in ((1, 0), (2, 0), (1, 1)):
        cv = p.Conv2D(2, 3, 3, stride, pad)
        k, bias = arr(cv.parameters()[0]), arr(cv.parameters()[1])
        x_np = spread(1 * 2 * 5 * 5).reshape(1, 2, 5, 5)
        got = arr(cv.forward(tensor([1, 2, 5, 5], x_np)))
        check("Conv2D stride=%d pad=%d" % (stride, pad),
              got, conv2d_ref(x_np, k, bias, stride, pad), atol=1e-4)

    x_np = spread(1 * 1 * 4 * 4).reshape(1, 1, 4, 4)
    mp = p.MaxPool2D(2)
    ref = x_np.reshape(1, 1, 2, 2, 2, 2).max(axis=(3, 5))
    check("MaxPool2D(2)", arr(mp.forward(tensor([1, 1, 4, 4], x_np))), ref)

    ap = p.AvgPool2D(2)
    ref = x_np.reshape(1, 1, 2, 2, 2, 2).mean(axis=(3, 5))
    check("AvgPool2D(2)", arr(ap.forward(tensor([1, 1, 4, 4], x_np))), ref)


# ------------------------------------------------------------------- recurrent
def test_rnn():
    print("\n=== RNN ===")
    rnn = p.RNN(3, 4)
    Wx, Wh, bias = (arr(t) for t in rnn.parameters())
    x_np = spread(5 * 1 * 3).reshape(5, 1, 3)

    h = np.zeros((1, 4))
    ref = []
    for t in range(5):
        h = np.tanh(x_np[t] @ Wx + h @ Wh + bias)
        ref.append(h.copy())
    ref = np.stack(ref)                      # [seq, batch, hidden]

    check("RNN output sequence",
          arr(rnn.forward(tensor([5, 1, 3], x_np))), ref, atol=1e-4)


# ----------------------------------------------------------------------- losses
def test_losses():
    print("\n=== losses ===")
    pr_np = spread(6).reshape(3, 2) * 0.5 + 0.1
    tg_np = spread(6)[::-1].reshape(3, 2) * 0.4
    pr, tg = tensor([3, 2], pr_np), tensor([3, 2], tg_np)

    check("mse_loss", arr(p.mse_loss(pr, tg)),
          np.array([np.mean((pr_np - tg_np) ** 2)]), atol=1e-5)
    check("mae_loss", arr(p.mae_loss(pr, tg)),
          np.array([np.mean(np.abs(pr_np - tg_np))]), atol=1e-5)

    d = 1.0
    r = pr_np - tg_np
    hub = np.where(np.abs(r) <= d, 0.5 * r ** 2, d * (np.abs(r) - 0.5 * d))
    check("huber_loss", arr(p.huber_loss(pr, tg)), np.array([hub.mean()]), atol=1e-5)

    # BCE on probabilities
    q_np = np.linspace(0.15, 0.85, 6).reshape(3, 2)
    y_np = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
    bce = -np.mean(y_np * np.log(q_np) + (1 - y_np) * np.log(1 - q_np))
    check("bce_loss", arr(p.bce_loss(tensor([3, 2], q_np), tensor([3, 2], y_np))),
          np.array([bce]), atol=1e-5)

    # sparse cross-entropy over a sequence, via log-sum-exp
    lg_np = spread(3 * 5).reshape(3, 5)
    targets = [0, 2, 4]
    lse = np.log(np.exp(lg_np - lg_np.max(axis=1, keepdims=True)).sum(axis=1)) \
        + lg_np.max(axis=1)
    ce = np.mean(lse - lg_np[np.arange(3), targets])
    check("cross_entropy_sparse_seq",
          arr(p.cross_entropy_sparse_seq(tensor([3, 5], lg_np), targets)),
          np.array([ce]), atol=1e-5)

    # KL divergence sum(p log(p/q))
    kp = np.linspace(0.1, 0.5, 6).reshape(3, 2)
    kq = np.linspace(0.5, 0.1, 6).reshape(3, 2)
    check("kl_divergence",
          arr(p.kl_divergence(tensor([3, 2], kp), tensor([3, 2], kq))),
          np.array([np.sum(kp * np.log(kp / kq))]), atol=1e-4)

    # regularisers
    w_np = spread(9).reshape(3, 3)
    w = tensor([3, 3], w_np)
    check("l1_regularization", arr(p.l1_regularization([w], 0.1)),
          np.array([0.1 * np.sum(np.abs(w_np))]), atol=1e-5)
    check("l2_regularization", arr(p.l2_regularization([w], 0.1)),
          np.array([0.1 * np.sum(w_np ** 2)]), atol=1e-5)


# -------------------------------------------------------------------- attention
def test_attention():
    print("\n=== MultiHeadAttention ===")
    for causal in (False, True):
        embed, heads, seq = 4, 2, 3
        mha = p.MultiHeadAttention(embed, heads, causal)
        Wq, Wk, Wv, Wo = (arr(t) for t in (mha.W_q, mha.W_k, mha.W_v, mha.W_o))
        x_np = spread(seq * embed).reshape(seq, embed)

        hd = embed // heads
        q, k, v = x_np @ Wq, x_np @ Wk, x_np @ Wv
        out = np.zeros((seq, embed))
        for h in range(heads):
            sl = slice(h * hd, (h + 1) * hd)
            scores = q[:, sl] @ k[:, sl].T / np.sqrt(hd)
            if causal:
                scores = np.where(np.triu(np.ones((seq, seq)), 1) > 0, -np.inf, scores)
            w = np.exp(scores - scores.max(axis=-1, keepdims=True))
            w = w / w.sum(axis=-1, keepdims=True)
            out[:, sl] = w @ v[:, sl]
        ref = out @ Wo

        check("MultiHeadAttention causal=%s" % causal,
              arr(mha.forward(tensor([seq, embed], x_np))), ref, atol=1e-4)


def test_lstm_gru():
    print("\n=== LSTM and GRU ===\n")
    # Standard formulations (as in the PyTorch docs), written from the maths:
    #   LSTM: i,f,o = sigmoid(..)   g = tanh(..)   c = f*c + i*g   h = o*tanh(c)
    #   GRU:  r,z = sigmoid(..)     n = tanh(x@Wxn + (r*h)@Whn + bn)
    #         h = (1-z)*n + z*h
    sig = lambda v: 1 / (1 + np.exp(-v))

    lstm = p.LSTM(3, 4)
    Wxi, Wxf, Wxo, Wxg, Whi, Whf, Who, Whg, bi, bf, bo, bg =         (arr(t) for t in lstm.parameters())
    x_np = spread(5 * 1 * 3).reshape(5, 1, 3)

    h = np.zeros((1, 4)); c = np.zeros((1, 4)); ref = []
    for t in range(5):
        xt = x_np[t]
        i = sig(xt @ Wxi + h @ Whi + bi)
        f = sig(xt @ Wxf + h @ Whf + bf)
        o = sig(xt @ Wxo + h @ Who + bo)
        g = np.tanh(xt @ Wxg + h @ Whg + bg)
        c = f * c + i * g
        h = o * np.tanh(c)
        ref.append(h.copy())
    check("LSTM output sequence",
          arr(lstm.forward(tensor([5, 1, 3], x_np))), np.stack(ref), atol=1e-4)

    gru = p.GRU(3, 4)
    Wxr, Wxz, Wxn, Whr, Whz, Whn, br, bz, bn = (arr(t) for t in gru.parameters())
    h = np.zeros((1, 4)); ref = []
    for t in range(5):
        xt = x_np[t]
        r = sig(xt @ Wxr + h @ Whr + br)
        z = sig(xt @ Wxz + h @ Whz + bz)
        n = np.tanh(xt @ Wxn + (r * h) @ Whn + bn)
        h = (1 - z) * n + z * h
        ref.append(h.copy())
    check("GRU output sequence",
          arr(gru.forward(tensor([5, 1, 3], x_np))), np.stack(ref), atol=1e-4)


def test_optimizers():
    print("\n=== optimizer update rules ===\n")
    g_np = spread(6).reshape(2, 3) * 0.3
    w0 = spread(6)[::-1].reshape(2, 3)

    def fresh():
        w = tensor([2, 3], w0)
        w.requires_grad = True
        w.grad = [float(v) for v in g_np.ravel()]
        return w

    # SGD: w <- w - lr * g
    w = fresh()
    opt = p.SGD([w], 0.1)
    opt.step()
    check("SGD step", arr(w), w0 - 0.1 * g_np, atol=1e-5)

    # Adam, one step from zero state. With m = (1-b1)g and v = (1-b2)g^2 the
    # bias correction makes mhat = g and vhat = g^2, so the first step is
    # w - lr * g / (|g| + eps), i.e. lr * sign(g) up to eps.
    lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
    w = fresh()
    opt = p.Adam([w], lr)
    opt.step()
    m = (1 - b1) * g_np
    v = (1 - b2) * g_np ** 2
    mhat = m / (1 - b1)
    vhat = v / (1 - b2)
    check("Adam first step", arr(w), w0 - lr * mhat / (np.sqrt(vhat) + eps), atol=1e-5)

    # two steps, to exercise the running moments and the step counter
    w = fresh()
    opt = p.Adam([w], lr)
    opt.step()
    w.grad = [float(x) for x in (g_np * 0.5).ravel()]
    opt.step()
    m2 = b1 * m + (1 - b1) * (g_np * 0.5)
    v2 = b2 * v + (1 - b2) * (g_np * 0.5) ** 2
    mh2 = m2 / (1 - b1 ** 2)
    vh2 = v2 / (1 - b2 ** 2)
    expect = w0 - lr * mhat / (np.sqrt(vhat) + eps)
    expect = expect - lr * mh2 / (np.sqrt(vh2) + eps)
    check("Adam second step", arr(w), expect, atol=1e-5)

    # AdamW: identical to Adam plus decoupled weight decay lr*wd*w
    wd = 0.05
    w = fresh()
    opt = p.AdamW([w], lr, 0.9, 0.999, 1e-8, wd)
    opt.step()
    adam_part = w0 - lr * mhat / (np.sqrt(vhat) + eps)
    check("AdamW step (decoupled decay)", arr(w), adam_part - lr * wd * w0, atol=1e-5)


def test_rmsprop_and_clip():
    print("\n=== RMSprop, gradient clipping, schedulers ===\n")
    g_np = spread(6).reshape(2, 3) * 0.3
    w0 = spread(6)[::-1].reshape(2, 3)

    def fresh(g=None):
        w = tensor([2, 3], w0)
        w.requires_grad = True
        w.grad = [float(v) for v in (g if g is not None else g_np).ravel()]
        return w

    # RMSprop: v <- beta*v + (1-beta)*g^2 ;  w <- w - lr*g/(sqrt(v)+eps)
    lr, beta, eps = 0.01, 0.9, 1e-8
    w = fresh()
    opt = p.RMSprop([w], lr, beta, eps)
    opt.step()
    v = (1 - beta) * g_np ** 2
    check("RMSprop first step", arr(w), w0 - lr * g_np / (np.sqrt(v) + eps), atol=1e-5)

    w = fresh()
    opt = p.RMSprop([w], lr, beta, eps)
    opt.step()
    g2 = g_np * 0.5
    w.grad = [float(x) for x in g2.ravel()]
    opt.step()
    v2 = beta * v + (1 - beta) * g2 ** 2
    expect = w0 - lr * g_np / (np.sqrt(v) + eps) - lr * g2 / (np.sqrt(v2) + eps)
    check("RMSprop second step", arr(w), expect, atol=1e-5)

    # SGD with momentum: buf <- m*buf + g ;  w <- w - lr*buf
    mom = 0.9
    w = fresh()
    opt = p.SGD([w], 0.1, mom)
    opt.step()
    buf = g_np.copy()
    step1 = w0 - 0.1 * buf
    w.grad = [float(x) for x in g2.ravel()]
    opt.step()
    buf = mom * buf + g2
    check("SGD with momentum, two steps", arr(w), step1 - 0.1 * buf, atol=1e-5)

    # clip_grad_norm scales gradients so the global L2 norm equals max_norm
    big = g_np * 50.0
    w = fresh(big)
    total = np.sqrt(np.sum(big ** 2))
    returned = p.clip_grad_norm([w], 1.0)
    check("clip_grad_norm returns the pre-clip norm",
          np.array([returned]), np.array([total]), atol=1e-3)
    check("clipped gradients scaled to max_norm",
          np.array(w.grad).reshape(2, 3), big / total, atol=1e-5)

    # below the threshold nothing should change
    w = fresh()
    small = np.sqrt(np.sum(g_np ** 2))
    p.clip_grad_norm([w], small * 10)
    check("clip_grad_norm leaves small gradients alone",
          np.array(w.grad).reshape(2, 3), g_np, atol=1e-6)

    # schedulers: step() advances and get_lr() must agree with it
    sl = p.StepLR(0.1, 3, 0.5)
    got = [sl.step() for _ in range(7)]
    check("StepLR schedule", np.array(got),
          np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]), atol=1e-6)

    el = p.ExponentialLR(0.1, 0.9)
    got = [el.step() for _ in range(5)]
    check("ExponentialLR schedule", np.array(got),
          np.array([0.1 * 0.9 ** i for i in range(5)]), atol=1e-6)
    check("ExponentialLR get_lr tracks step",
          np.array([el.get_lr()]), np.array([got[-1]]), atol=1e-9)

    cl = p.CosineAnnealingLR(0.1, 6, 0.01)
    got = [cl.step() for _ in range(6)]
    ref = [0.01 + (0.1 - 0.01) * (1 + np.cos(np.pi * i / 6)) / 2 for i in range(6)]
    check("CosineAnnealingLR schedule", np.array(got), np.array(ref), atol=1e-5)
    check("CosineAnnealingLR get_lr tracks step",
          np.array([cl.get_lr()]), np.array([got[-1]]), atol=1e-9)


def test_metrics():
    print("\n=== metrics ===\n")
    # 8 samples, binary, threshold 0.5
    pr = np.array([0.9, 0.2, 0.8, 0.4, 0.6, 0.1, 0.7, 0.3])
    yt = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    P = tensor([8, 1], pr)
    Y = tensor([8, 1], yt)

    pred = (pr >= 0.5).astype(float)

    # confusion matrix, rows = true class, columns = predicted class
    cm = np.zeros((2, 2))
    for t, q in zip(yt.astype(int), pred.astype(int)):
        cm[t, q] += 1
    check("confusion_matrix", arr(p.confusion_matrix(P, Y)), cm, atol=0)

    check("accuracy", np.array([p.accuracy(P, Y)]),
          np.array([np.mean(pred == yt)]), atol=1e-6)

    # precision and recall are macro-averaged over both classes, so the
    # reference averages the per-class figures rather than taking the
    # positive class alone.
    prec_c, rec_c = [], []
    for c in range(2):
        tp_c = cm[c, c]
        fp_c = cm[:, c].sum() - tp_c
        fn_c = cm[c, :].sum() - tp_c
        prec_c.append(tp_c / (tp_c + fp_c) if tp_c + fp_c else 0.0)
        rec_c.append(tp_c / (tp_c + fn_c) if tp_c + fn_c else 0.0)
    pm, rm = float(np.mean(prec_c)), float(np.mean(rec_c))

    check("precision (macro)", np.array([p.precision(P, Y)]), np.array([pm]), atol=1e-6)
    check("recall (macro)",    np.array([p.recall(P, Y)]),    np.array([rm]), atol=1e-6)
    # f1 here is the harmonic mean of the macro averages, not the mean of the
    # per-class F1 scores; see the note in metrics.hpp
    check("f1_score (of the macro averages)", np.array([p.f1_score(P, Y)]),
          np.array([2 * pm * rm / (pm + rm)]), atol=1e-6)

    # r2 on a regression target
    yp = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
    yy = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    ss_res = np.sum((yy - yp) ** 2)
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    check("r2_score", np.array([p.r2_score(tensor([5, 1], yp), tensor([5, 1], yy))]),
          np.array([1 - ss_res / ss_tot]), atol=1e-5)


def test_wider_shapes():
    print("\n=== larger shapes and batches ===\n")
    # Linear with a bigger batch
    fc = p.Linear(7, 5)
    W, b = arr(fc.weights), arr(fc.bias)
    x = spread(16 * 7).reshape(16, 7)
    check("Linear batch 16", arr(fc.forward(tensor([16, 7], x))), x @ W + b, atol=1e-4)

    # Conv2D with a real batch and several channels
    cv = p.Conv2D(3, 4, 3, 1, 1)
    k, bias = arr(cv.parameters()[0]), arr(cv.parameters()[1])
    xi = spread(4 * 3 * 8 * 8).reshape(4, 3, 8, 8)
    check("Conv2D batch 4, 3->4 channels, 8x8",
          arr(cv.forward(tensor([4, 3, 8, 8], xi))),
          conv2d_ref(xi, k, bias, 1, 1), atol=1e-3)

    # recurrent layers with batch > 1 and a longer sequence
    sig = lambda v: 1 / (1 + np.exp(-v))
    rnn = p.RNN(4, 6)
    Wx, Wh, rb = (arr(t) for t in rnn.parameters())
    xs = spread(12 * 3 * 4).reshape(12, 3, 4)
    h = np.zeros((3, 6)); ref = []
    for t in range(12):
        h = np.tanh(xs[t] @ Wx + h @ Wh + rb)
        ref.append(h.copy())
    check("RNN seq 12, batch 3",
          arr(rnn.forward(tensor([12, 3, 4], xs))), np.stack(ref), atol=1e-4)

    lstm = p.LSTM(4, 6)
    Wxi, Wxf, Wxo, Wxg, Whi, Whf, Who, Whg, bi, bf, bo, bg =         (arr(t) for t in lstm.parameters())
    h = np.zeros((3, 6)); c = np.zeros((3, 6)); ref = []
    for t in range(12):
        xt = xs[t]
        i = sig(xt @ Wxi + h @ Whi + bi)
        f = sig(xt @ Wxf + h @ Whf + bf)
        o = sig(xt @ Wxo + h @ Who + bo)
        g = np.tanh(xt @ Wxg + h @ Whg + bg)
        c = f * c + i * g
        h = o * np.tanh(c)
        ref.append(h.copy())
    check("LSTM seq 12, batch 3",
          arr(lstm.forward(tensor([12, 3, 4], xs))), np.stack(ref), atol=1e-4)

    # attention with 4 heads over a longer sequence
    embed, heads, seq = 8, 4, 7
    mha = p.MultiHeadAttention(embed, heads, True)
    Wq, Wk, Wv, Wo = (arr(t) for t in (mha.W_q, mha.W_k, mha.W_v, mha.W_o))
    xa = spread(seq * embed).reshape(seq, embed)
    hd = embed // heads
    q, k, v = xa @ Wq, xa @ Wk, xa @ Wv
    out = np.zeros((seq, embed))
    for hh in range(heads):
        sl = slice(hh * hd, (hh + 1) * hd)
        sc = q[:, sl] @ k[:, sl].T / np.sqrt(hd)
        sc = np.where(np.triu(np.ones((seq, seq)), 1) > 0, -np.inf, sc)
        wt = np.exp(sc - sc.max(axis=-1, keepdims=True))
        wt = wt / wt.sum(axis=-1, keepdims=True)
        out[:, sl] = wt @ v[:, sl]
    check("MultiHeadAttention 4 heads, seq 7, causal",
          arr(mha.forward(tensor([seq, embed], xa))), out @ Wo, atol=1e-4)


def test_batchnorm_modes():
    print("\n=== BatchNorm training vs inference ===\n")
    bn = p.BatchNorm(3, 1e-5, True, 0.1)
    g, b = arr(bn.parameters()[0]), arr(bn.parameters()[1])
    X = spread(12).reshape(4, 3)

    # training: batch statistics, and the running estimate is folded in
    out = arr(bn.forward(tensor([4, 3], X)))
    mu, var = X.mean(axis=0), X.var(axis=0)
    check("BatchNorm train uses batch stats",
          out, g * (X - mu) / np.sqrt(var + 1e-5) + b, atol=1e-4)

    rm = 0.9 * 0.0 + 0.1 * mu          # running_mean starts at 0
    rv = 0.9 * 1.0 + 0.1 * var         # running_var starts at 1
    check("running_mean after one batch", np.array(bn.running_mean), rm, atol=1e-6)
    check("running_var after one batch",  np.array(bn.running_var),  rv, atol=1e-6)

    # inference: running estimate, and no further updates
    bn.eval()
    out = arr(bn.forward(tensor([4, 3], X)))
    check("BatchNorm eval uses running stats",
          out, g * (X - rm) / np.sqrt(rv + 1e-5) + b, atol=1e-4)
    # running stats are stored as float32, so compare at float32 precision
    check("eval leaves running_mean untouched", np.array(bn.running_mean), rm, atol=1e-6)

    # a single sample has zero batch variance, so this only works in eval mode
    one = spread(3).reshape(1, 3)
    out = arr(bn.forward(tensor([1, 3], one)))
    check("single-sample inference",
          out, g * (one - rm) / np.sqrt(rv + 1e-5) + b, atol=1e-4)


def test_batching():
    print("\n=== batching ===\n")
    # The decisive property: a batch must give exactly what the sequences give
    # one at a time. Attention is the only layer that must not see across a
    # sequence boundary, so a mistake there shows up here and nowhere else.
    p.Tensor.manual_seed(7)
    E, H, S, B = 8, 2, 5, 4

    for causal in (False, True):
        for rope in (False, True):
            mha = p.MultiHeadAttention(E, H, causal, rope)
            seqs = [[0.1 * ((i * 7 + b * 3) % 11 - 5) for i in range(S * E)]
                    for b in range(B)]

            alone = []
            for row in seqs:
                alone += list(mha.forward(tensor([S, E], row)).data)

            flat = [v for row in seqs for v in row]
            together = list(mha.forward(tensor([B * S, E], flat), B).data)

            check("attention causal=%s rope=%s: batched == per-sequence"
                  % (causal, rope),
                  np.array(together), np.array(alone), atol=1e-6)

    # and the same for the whole model, which also exercises the position
    # embedding restarting at each sequence
    V, L = 50, 2
    for rope in (False, True):
        gpt = p.GPT(V, S + 1, E, H, L, 0, rope)
        ids = [[(i * 7 + b * 3) % V for i in range(S)] for b in range(B)]

        alone = []
        for row in ids:
            alone += list(gpt.forward(tensor([S], row)).data)
        flat = [t for row in ids for t in row]
        together = list(gpt.forward(tensor([B * S], flat), B).data)

        check("GPT rope=%s: batched logits == per-sequence" % rope,
              np.array(together), np.array(alone), atol=1e-4)

    # gradients: the batched loss averages over B*S positions where the loop
    # averages over S each, so the batched gradient is the per-sequence sum
    # divided by B
    gpt = p.GPT(V, S + 1, E, H, L)
    ids = [[(i * 5 + b) % V for i in range(S)] for b in range(B)]
    tgts = [[(i * 5 + b + 1) % V for i in range(S)] for b in range(B)]

    for t in gpt.parameters():
        t.reset_grad()
    for row, tg in zip(ids, tgts):
        p.cross_entropy_sparse_seq(gpt.forward(tensor([S], row)), tg).backward()
    alone = [list(t.grad) for t in gpt.parameters()]

    for t in gpt.parameters():
        t.reset_grad()
    flat = [x for row in ids for x in row]
    tflat = [x for tg in tgts for x in tg]
    p.cross_entropy_sparse_seq(gpt.forward(tensor([B * S], flat), B),
                               tflat).backward()
    together = [list(t.grad) for t in gpt.parameters()]

    # The two routes multiply matrices of different shapes -- 20 rows at once
    # against 5 at a time -- and a BLAS library is free to pick a different
    # kernel, and so a different summation order, for each. Under MKL that
    # moves gradients by a few parts in a million of their scale.
    #
    # The scale is the tensor's, not the element's: a gradient that is a sum
    # of large terms which nearly cancel carries rounding error the size of
    # the terms, so an element-wise relative test fails on it while the
    # tensor is right. So each parameter's worst error is measured against
    # its largest gradient. A real batching bug is off by orders of magnitude
    # more than the 1e-5 allowed here.
    worst = 0.0
    for A, Bg in zip(alone, together):
        a = np.array(A) / B
        b = np.array(Bg)
        scale = max(float(np.max(np.abs(a))), 1e-12)
        worst = max(worst, float(np.max(np.abs(a - b))) / scale)
    check("GPT batched gradients == per-sequence sum / B (rel. to scale)",
          np.array([worst]), np.array([0.0]), atol=1e-5)

    # a batch that does not divide the token count must be refused
    threw = False
    try:
        gpt.forward(tensor([S * 3], [1.0] * (S * 3)), 2)
    except Exception:
        threw = True
    check("an indivisible batch raises", np.array([1.0 if threw else 0.0]),
          np.array([1.0]), atol=0)


def test_views():
    print("\n=== strided views ===\n")
    A = spread(12).reshape(3, 4)
    a = tensor([3, 4], A)

    v = p.transpose_view(a)
    check("transpose_view shape", np.array(v.shape), np.array([4, 3]), atol=0)
    check("transpose_view is a view",
          np.array([1.0 if v.is_view() else 0.0]), np.array([1.0]), atol=0)
    check("and is non-contiguous",
          np.array([0.0 if v.is_contiguous() else 1.0]), np.array([1.0]), atol=0)
    check("materialises to numpy .T", arr(p.contiguous(v)), A.T)

    r = p.reshape_view(a, [4, 3])
    check("reshape_view stays contiguous",
          np.array([1.0 if r.is_contiguous() else 0.0]), np.array([1.0]), atol=0)
    check("reshape_view materialises correctly", arr(p.contiguous(r)), A.reshape(4, 3))

    # transposing twice must return the original
    check("transpose_view twice is the identity",
          arr(p.contiguous(p.transpose_view(p.transpose_view(a)))), A)

    # contiguous() on an owned tensor must not copy
    check("contiguous() is a no-op on owned storage",
          np.array([0.0 if p.contiguous(a).is_view() else 1.0]),
          np.array([1.0]), atol=0)

    # The gradient convention: a view's grad is row-major over the view's own
    # shape, not indexed through the strides. Getting this wrong was a real
    # defect -- one backward wrote in storage order while another read
    # row-major, which disagreed silently wherever it did not crash. The test
    # is that the view route gives exactly what the copying route gives.
    x = tensor([3, 4], spread(12))
    x.requires_grad = True
    x.reset_grad()
    y = p.contiguous(p.transpose_view(x))
    p.mean(p.multiply(y, y), -1).backward()
    via_view = list(x.grad)

    x.reset_grad()
    z = p.transpose_op(x)
    p.mean(p.multiply(z, z), -1).backward()
    via_copy = list(x.grad)

    check("view gradients equal transpose_op gradients",
          np.array(via_view), np.array(via_copy), atol=1e-6)

    # a non-contiguous view cannot be reshaped without materialising first
    threw = False
    try:
        p.reshape_view(p.transpose_view(a), [12])
    except Exception:
        threw = True
    check("reshaping a non-contiguous view raises",
          np.array([1.0 if threw else 0.0]), np.array([1.0]), atol=0)


def main():
    print("Forward-pass verification against numpy reference implementations")
    test_ops()
    test_linear()
    test_embedding_flatten()
    test_norms()
    test_batchnorm_modes()
    test_vision()
    test_rnn()
    test_lstm_gru()
    test_losses()
    test_attention()
    test_batching()
    test_views()
    test_optimizers()
    test_rmsprop_and_clip()
    test_metrics()
    test_wider_shapes()

    print("\n%d comparisons, %d failed" % (CHECKS[0], len(FAILURES)))
    if FAILURES:
        print("\nFailures:")
        for f in FAILURES:
            print("  -", f)
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
