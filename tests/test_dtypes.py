"""Dtypes: promotion, integer and bool ops, float64 compute, half precision, autocast.

References are numpy, computed independently; PyTorch's promotion table is
used as well when torch is installed.
"""
import sys

import numpy as np
import prometheus as p

try:
    import torch
except ImportError:
    torch = None

CHECKS, FAILURES = [0], []


def check(name, ok, detail=""):
    CHECKS[0] += 1
    if not ok:
        FAILURES.append(name)
    print("  %-62s %s %s" % (name, "ok" if ok else "** FAIL **", detail))


def T(values, dtype=p.float32, shape=None):
    a = np.asarray(values)
    return p.Tensor(list(shape or a.shape), [float(v) for v in a.reshape(-1)], dtype)


def A(t):
    return np.array(t.data, dtype=np.float64).reshape(list(t.shape))


def raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


ALL = ["bool", "int32", "int64", "float16", "bfloat16", "float32", "float64"]


def test_promotion():
    print("\n=== promotion ===\n")
    expect = {("int32", "int64"): "int64", ("bool", "int32"): "int32", ("int64", "float16"): "float16",
              ("float16", "bfloat16"): "float32", ("float32", "float64"): "float64",
              ("bool", "float16"): "float16", ("bfloat16", "float32"): "float32"}
    for (x, y), want in expect.items():
        got = p.promote_types(getattr(p, x), getattr(p, y))
        check("promote %s + %s -> %s" % (x, y, want), got == getattr(p, want), str(got))
    if torch is not None:
        bad = []
        for x in ALL:
            for y in ALL:
                ours = str(p.promote_types(getattr(p, x), getattr(p, y))).split(".")[-1]
                theirs = str(torch.promote_types(getattr(torch, x), getattr(torch, y))).split(".")[-1]
                if ours != theirs:
                    bad.append((x, y, ours, theirs))
        check("every pair matches torch.promote_types (49)", not bad, str(bad[:3]))

    i32 = T([1, 2, 3], p.int32)
    check("int32 tensor + 1 stays int32", (i32 + 1).dtype == p.int32)
    check("int32 tensor + 1.5 is float32", (i32 + 1.5).dtype == p.float32)
    check("int32 / int32 is true division, float32", np.allclose(A(i32 / 2), [0.5, 1, 1.5]) and
          (i32 / 2).dtype == p.float32)
    check("bool tensor + 1 is int64", (T([1, 0], p.bool) + 1).dtype == p.int64)
    f16 = T([1, 2], p.float16)
    check("float16 tensor * 2.5 stays float16", (f16 * 2.5).dtype == p.float16)
    check("float16 + float32 tensor is float32", (f16 + T([1, 2])).dtype == p.float32)
    check("float32 + float64 tensor is float64", (T([1, 2]) + T([1, 2], p.float64)).dtype == p.float64)
    check("0-d int64 does not widen an int32 tensor",
          p.result_type(i32, p.scalar_tensor(3, p.int64)) == p.int32)


def test_integer_and_bool():
    print("\n=== integer and bool ops ===\n")
    big = 2 ** 40 + 7
    a = T([big, -big, 5], p.int64)
    b = T([3, 3, -2], p.int64)
    na, nb = np.array([big, -big, 5], np.int64), np.array([3, 3, -2], np.int64)
    check("int64 add exact past 2^24", np.array_equal(np.array((a + b).numpy()), na + nb))
    check("int64 multiply exact", np.array_equal(np.array((a * b).numpy()), na * nb))
    check("floor_divide is Python //", np.array_equal(np.array((a // b).numpy()), na // nb))
    check("remainder takes the divisor's sign", np.array_equal(np.array((a % b).numpy()), na % nb))
    check("integer division by zero raises", raises(lambda: a // T([0, 1, 1], p.int64)))
    check("sum of int64 is int64 and exact",
          p.sum(a, -1).dtype == p.int64 and int(p.sum(a, -1).item()) == int(na.sum()))
    check("sum of bool counts", int(p.sum(T([1, 0, 1, 1], p.bool), -1).item()) == 3)
    check("relu / abs keep int dtype", p.relu(a).dtype == p.int64 and p.abs_op(a).dtype == p.int64)
    check("exp of int is float32", p.exp_op(T([0, 1], p.int32)).dtype == p.float32)

    x = np.array([[1.0, -2.0, 3.0], [0.5, 0.0, -1.0]])
    t = T(x)
    for name, op, ref in (("lt", lambda u: u < 0.5, x < 0.5), ("le", lambda u: u <= 0.5, x <= 0.5),
                          ("gt", lambda u: u > 0, x > 0), ("ge", lambda u: u >= 0, x >= 0),
                          ("eq", lambda u: u == 0.5, x == 0.5), ("ne", lambda u: u != 3, x != 3)):
        r = op(t)
        check("compare %s: bool, matches numpy" % name, r.dtype == p.bool and np.array_equal(A(r) != 0, ref))
    m1, m2 = t > 0, t < 2
    check("logical and / or / xor / not",
          np.array_equal(A(m1 & m2) != 0, (x > 0) & (x < 2)) and
          np.array_equal(A(m1 | m2) != 0, (x > 0) | (x < 2)) and
          np.array_equal(A(m1 ^ m2) != 0, (x > 0) ^ (x < 2)) and
          np.array_equal(A(~m1) != 0, ~(x > 0)))
    check("any / all", bool(p.any(m1).item()) and not bool(p.all(m1).item()))
    row = T([1.0, 0.0, 2.0], shape=[1, 3])
    check("comparison broadcasts", list((t > row).shape) == [2, 3] and
          np.array_equal(A(t > row) != 0, x > np.array([[1.0, 0.0, 2.0]])))
    check("isnan / isinf / isfinite",
          np.array_equal(A(p.isnan(T([np.nan, 1.0, np.inf]))) != 0, [True, False, False]) and
          np.array_equal(A(p.isinf(T([np.nan, 1.0, np.inf]))) != 0, [False, False, True]) and
          np.array_equal(A(p.isfinite(T([np.nan, 1.0, np.inf]))) != 0, [False, True, False]))
    check("bool - bool raises, as in PyTorch", raises(lambda: m1 - m2))

    # where / masked_fill, values and gradients
    a = T(x); a.requires_grad = True
    b = T(-x); b.requires_grad = True
    w = p.where(t > 0, a, b)
    check("where picks element-wise", np.allclose(A(w), np.where(x > 0, x, -x)))
    p.sum(p.multiply(w, T(np.arange(6.0).reshape(2, 3))), -1).backward()
    wts = np.arange(6.0).reshape(2, 3)
    check("where: gradient reaches a where chosen", np.allclose(np.reshape(a.grad, (2, 3)), np.where(x > 0, wts, 0)))
    check("where: and b elsewhere", np.allclose(np.reshape(b.grad, (2, 3)), np.where(x > 0, 0, wts)))
    c = T(x); c.requires_grad = True
    mf = p.masked_fill(c, t < 0, -1e9)
    check("masked_fill values", np.allclose(A(mf), np.where(x < 0, -1e9, x)))
    p.sum(mf, -1).backward()
    check("masked_fill: zero gradient where filled", np.allclose(np.reshape(c.grad, (2, 3)), (x >= 0) * 1.0))
    scores = T(np.zeros((2, 3)))
    causal = T(np.triu(np.ones((2, 3)), 1), p.bool)
    sm = p.softmax(p.masked_fill(scores, causal, float("-inf")))
    check("causal mask with -inf gives a clean softmax",
          np.allclose(A(sm), [[1, 0, 0], [0.5, 0.5, 0]]))


def test_float64():
    print("\n=== float64: computed in double, gradients in double ===\n")
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2.0, (3, 4))
    y = rng.uniform(0.5, 2.0, (4,))
    X, Y = T(x, p.float64), T(y, p.float64)
    cases = [
        ("add", lambda a, b: a + b, x + y), ("subtract", lambda a, b: a - b, x - y),
        ("multiply", lambda a, b: a * b, x * y), ("divide", lambda a, b: a / b, x / y),
        ("exp", lambda a, b: p.exp_op(a), np.exp(x)), ("log", lambda a, b: p.log_op(a), np.log(x)),
        ("tanh", lambda a, b: p.tanh_op(a), np.tanh(x)), ("sqrt", lambda a, b: p.sqrt_op(a), np.sqrt(x)),
        ("sigmoid", lambda a, b: p.sigmoid(a), 1 / (1 + np.exp(-x))),
        ("softmax", lambda a, b: p.softmax(a), np.exp(x) / np.exp(x).sum(-1, keepdims=True)),
        ("matmul", lambda a, b: p.matmul(a, p.reshape(b, [4, 1])), x @ y.reshape(4, 1)),
        ("sum dim 0", lambda a, b: p.sum(a, 0), x.sum(0)), ("mean all", lambda a, b: p.mean(a, -1), [x.mean()]),
    ]
    for name, fn, ref in cases:
        r = fn(X, Y)
        err = np.abs(A(r) - np.asarray(ref)).max()
        check("forward %-10s float64, error %.1e" % (name, err), r.dtype == p.float64 and err < 1e-13)

    # gradients against finite differences at float64 precision
    def gc(name, fn, inputs):
        r = p.gradcheck(fn, inputs, eps=1e-4, atol=1e-9, rtol=1e-7, max_points=6)
        check("gradcheck float64 %-26s" % name, r["passed"], "max abs err %.1e" % r["max_abs_err"])

    def leaf(v):
        t = T(v, p.float64)
        t.requires_grad = True
        return t
    a, b = leaf(x), leaf(y)
    gc("broadcast a*b/(a+b)", lambda: p.divide(p.multiply(a, b), p.add(a, b)), [a, b])
    gc("tanh(exp) - log", lambda: p.subtract(p.tanh_op(p.exp_op(a)), p.log_op(a)), [a])
    gc("softmax * weights", lambda: p.multiply(p.softmax(a), T(np.arange(12.0).reshape(3, 4), p.float64)), [a])
    m1, m2 = leaf(rng.normal(size=(2, 3, 4))), leaf(rng.normal(size=(4, 5)))
    gc("batched matmul", lambda: p.matmul(m1, m2), [m1, m2])
    gc("gelu, sigmoid, sqrt", lambda: p.add(p.gelu(a), p.multiply(p.sigmoid(a), p.sqrt_op(a))), [a])
    gc("concat + sum(dim) + max", lambda: p.add(p.sum(p.concat_op([a, a], 0), 1), p.max_op(a)), [a])
    bias = leaf(rng.normal(size=(5,)))
    gc("linear_op", lambda: p.linear_op(m1, m2, bias), [m1, m2, bias])

    # .grad in double: d/dx of x^3 at x = 1/3 is exactly 1/3 to double precision
    z = leaf(np.array([1.0 / 3.0]))
    p.sum(p.multiply(p.multiply(z, z), z), -1).backward()
    check(".grad is held in double", abs(z.grad[0] - 1.0 / 3.0) < 1e-15, "%.17g" % z.grad[0])

    # second derivative through autograd.grad, in double
    w = leaf(np.array([0.7, -1.3]))
    out = p.sum(p.tanh_op(w), -1)
    (g,) = p.grad(out, [w], create_graph=True)
    (h,) = p.grad(p.sum(g, -1), [w])
    tw = np.tanh(np.array([0.7, -1.3]))
    check("second derivative in float64", np.abs(A(h) - (-2 * tw * (1 - tw ** 2))).max() < 1e-14)
    lin = p.Linear(4, 2).forward(T(x, p.float64))
    check("Linear runs in float64 (composed from ops)", lin.dtype == p.float64)
    check("a fused layer refuses float64 by name",
          raises(lambda: p.Conv2D(1, 1, 3, 1, 1).forward(T(np.zeros((1, 1, 4, 4)), p.float64))))


def test_half():
    print("\n=== float16 / bfloat16: stored narrow, computed in float32 ===\n")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 8)).astype(np.float32)
    for dt, npt in (("float16", np.float16), ("bfloat16", None)):
        d = getattr(p, dt)
        X = T(x, d)
        r = p.gelu(p.matmul(X, p.transpose(X)))
        check("%s: matmul + gelu return %s" % (dt, dt), r.dtype == d)
        xr = A(X)        # the stored values
        ref = xr @ xr.T
        ref = 0.5 * ref * (1 + np.tanh(0.7978845608 * (ref + 0.044715 * ref ** 3)))
        rel = np.abs(A(r) - ref).max() / np.abs(ref).max()
        check("%s: within its precision of float32 compute (%.1e)" % (dt, rel), rel < (1e-3 if dt == "float16" else 8e-3))
        leafx = T(x, d)
        leafx.requires_grad = True
        p.sum(p.multiply(leafx, leafx), -1).backward()
        check("%s: gradient reaches a %s leaf" % (dt, dt),
              np.allclose(np.reshape(leafx.grad, x.shape), 2 * A(leafx), rtol=1e-6))
        out = p.Linear(8, 3).forward(T(x, d))
        check("%s: layers accept it (computed in float32)" % dt, list(out.shape) == [4, 3])


def test_autocast():
    print("\n=== autocast and GradScaler ===\n")
    rng = np.random.default_rng(2)
    p.Tensor.manual_seed(0)
    lin = p.Linear(16, 8)
    x = T(rng.normal(size=(4, 16)))
    with p.autocast(p.bfloat16):
        h = lin.forward(x)
        s = p.softmax(h)
        loss = p.mean(p.sum(p.multiply(s, s), 1), -1)
    check("inside autocast linear returns bfloat16", h.dtype == p.bfloat16)
    check("softmax under autocast returns float32", s.dtype == p.float32)
    check("autocast is off after the block", p.autocast_dtype() == p.float32)
    ref = A(x) @ np.array(lin.parameters()[0].data).reshape(16, 8) + np.array(lin.parameters()[1].data)
    check("bfloat16 linear within bfloat16 precision", np.abs(A(h) - ref).max() / np.abs(ref).max() < 1e-2)
    loss.backward()
    gw = np.array(lin.parameters()[0].grad)
    check("float32 weights get finite float32 gradients", gw.size == 128 and np.isfinite(gw).all() and np.abs(gw).max() > 0)

    lin.parameters()[0].zero_grad(); lin.parameters()[1].zero_grad()
    with p.autocast(p.float16):
        loss16 = p.mean(p.sum(p.multiply(lin.forward(x), lin.forward(x)), 1), -1)
    scaler = p.GradScaler()
    p.Tensor.backward(scaler.scale_loss(loss16))
    clean = scaler.unscale(lin.parameters())
    g16 = np.array(lin.parameters()[0].grad)
    check("GradScaler: scaled float16 backward unscales to finite grads", clean and np.isfinite(g16).all())


def main():
    print("Dtypes")
    test_promotion()
    test_integer_and_bool()
    test_float64()
    test_half()
    test_autocast()
    print("\n%d checks, %d failed" % (CHECKS[0], len(FAILURES)))
    print("PASS" if not FAILURES else "FAIL")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
