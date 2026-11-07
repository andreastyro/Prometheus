"""
Python API checks: the bindings, gradcheck from Python, and the Python GPT.

The C++ suite covers the library itself. This file covers the surface a Python
user actually touches, and asserts the two models agree.

Run directly:  python tests/test_python_api.py
"""

import os
import sys

import prometheus as p

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from python.models import MLP          # noqa: E402

FAILURES = []
CHECKS = [0]


def check(name, cond, detail=""):
    CHECKS[0] += 1
    if not cond:
        FAILURES.append(name + ((" (" + detail + ")") if detail else ""))
    print("  %-52s %s" % (name, "ok" if cond else "** FAIL **"))


def tensor(shape, values):
    t = p.Tensor(shape)
    t.data = [float(v) for v in values]
    return t


def spread(n):
    return [(-0.9 + 1.8 * i / max(1, n - 1)) + 0.013 for i in range(n)]


def test_bindings_present():
    print("\n=== bindings ===")
    for name in ("Tensor", "Linear", "Conv2D", "LayerNorm", "GroupNorm", "BatchNorm",
                 "Embedding", "MultiHeadAttention", "TransformerBlock", "GPT",
                 "RNN", "LSTM", "GRU", "Sequential", "Flatten", "Dropout",
                 "ResidualBlock", "Adam", "AdamW", "SGD", "RMSprop",
                 "gradcheck", "gelu", "transpose_op", "reshape_op"):
        check("exposed: " + name, hasattr(p, name))

    check("GPT is a Module subclass", issubclass(p.GPT, p.Module))
    check("Tensor.manual_seed exists", hasattr(p.Tensor, "manual_seed"))


def test_reproducibility():
    print("\n=== reproducibility ===")
    p.Tensor.manual_seed(1234)
    a = p.Tensor.randn([40]).data
    p.Tensor.manual_seed(1234)
    b = p.Tensor.randn([40]).data
    check("manual_seed reproduces randn", a == b)

    p.Tensor.manual_seed(99)
    a = p.Tensor.randn([40]).data
    check("different seed gives different values", a != b)

    # Dropout draws from the same generator, so it is reproducible too
    x = tensor([4, 4], spread(16))
    x.requires_grad = True
    d = p.Dropout(0.5)
    p.Tensor.manual_seed(7)
    m1 = d.forward(x).data
    p.Tensor.manual_seed(7)
    m2 = d.forward(x).data
    check("manual_seed reproduces the dropout mask", m1 == m2)


def test_gradcheck_from_python():
    print("\n=== gradcheck through the bindings ===")
    x = tensor([2, 4], spread(8))
    x.requires_grad = True

    fc = p.Linear(4, 3)
    r = p.gradcheck(lambda: fc.forward(x), [fc.weights, fc.bias, x])
    check("Linear gradients", r["passed"], r["report"][:120])
    check("gradcheck reports a check count", r["checked"] > 0)

    r = p.gradcheck(lambda: p.gelu(x), [x])
    check("gelu gradients", r["passed"], r["report"][:120])

    gn = p.GroupNorm(2, 4)
    xg = tensor([1, 4, 2, 2], spread(16))
    xg.requires_grad = True
    r = p.gradcheck(lambda: gn.forward(xg), gn.parameters() + [xg])
    check("GroupNorm gradients", r["passed"], r["report"][:120])

    bn = p.BatchNorm(4)
    xb = tensor([3, 4], spread(12))
    xb.requires_grad = True
    r = p.gradcheck(lambda: bn.forward(xb), bn.parameters() + [xb])
    check("BatchNorm gradients", r["passed"], r["report"][:120])

    # a deliberately detached tensor must be reported as failing, so we know
    # gradcheck is capable of saying no
    det = tensor([2, 2], [0.3, -0.4, 0.5, 0.2])
    det.requires_grad = True
    r = p.gradcheck(lambda: det.transpose(), [det])   # transpose() detaches by design
    check("gradcheck rejects a detached op", not r["passed"])


def test_python_gpt_matches_cpp():
    print("\n=== Python GPT vs C++ GPT ===")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python.gpt import GPT

    V, L, E, H, N = 32, 16, 8, 2, 2
    p.Tensor.manual_seed(4)
    cpp = p.GPT(V, L, E, H, N)
    py = GPT(V, L, E, H, N)

    cp, pp = cpp.parameters(), py.parameters()
    check("same parameter count", len(cp) == len(pp),
          "cpp=%d py=%d" % (len(cp), len(pp)))
    check("same parameter shapes", all(a.shape == b.shape for a, b in zip(cp, pp)))

    for a, b in zip(cp, pp):
        b.data = list(a.data)

    ids = [1, 2, 3, 4, 5]
    lc = cpp.forward(tensor([len(ids)], ids))
    lp = py.forward(ids)
    check("same logit shape", lc.shape == lp.shape)
    diff = max(abs(u - v) for u, v in zip(lc.data, lp.data))
    check("logits agree to 1e-4", diff < 1e-4, "max diff %.3e" % diff)

    loss = p.cross_entropy_sparse_seq(lp, [2, 3, 4, 5, 6])
    loss.backward()
    tied = sum(abs(v) for v in py.tok_emb.weight.grad)
    check("weight tying delivers gradient to tok_emb", tied > 0)
    trained = sum(1 for t in pp if any(v != 0.0 for v in t.grad))
    check("every parameter receives gradient", trained == len(pp),
          "%d of %d" % (trained, len(pp)))


def test_save_load():
    print("\n=== save / load round trip ===")
    model = p.Sequential([p.Linear(3, 4), p.ReLU(), p.Linear(4, 1)])
    before = [list(t.data) for t in model.parameters()]
    path = "_test_python_api.bin"
    p.save_model(path, model.parameters())
    loaded = p.load_model(path)
    check("parameter count preserved", len(loaded) == len(before))
    check("values bit-exact after reload",
          all(list(t.data) == b for t, b in zip(loaded, before)))
    os.remove(path)

    threw = False
    try:
        p.load_model("_definitely_missing.bin")
    except Exception:
        threw = True
    check("loading a missing file raises", threw)


def test_lazy_grad_and_no_grad():
    print("\n=== lazy gradients and no_grad ===")
    # the buffer used to be sized in the constructor, doubling the memory of
    # every tensor including constants and inference activations
    a = p.Tensor([256, 256])
    check("a fresh tensor allocates no gradient buffer", len(a.grad) == 0,
          "%d entries" % len(a.grad))
    a.requires_grad = True
    check("marking requires_grad allocates it", len(a.grad) == a.num_el())

    m = MLP(8, 4, (16,))
    x = tensor([4, 8], spread(32))
    x.requires_grad = True

    with_grad = m.forward(x)
    check("a training forward records a graph", with_grad.requires_grad)
    check("and allocates a gradient buffer", len(with_grad.grad) == with_grad.num_el())

    import python.models as models
    with models.no_grad():
        inferred = m.forward(x)
    check("no_grad forward records no graph", not inferred.requires_grad)
    check("and allocates no gradient buffer", len(inferred.grad) == 0)
    check("values are identical either way",
          list(with_grad.data) == list(inferred.data))
    check("grad mode is restored", p.is_grad_enabled())

    # backward on a graph-less result must raise rather than silently leave
    # every parameter at zero, which is indistinguishable from a severed graph
    with models.no_grad():
        y = m.forward(x)
    threw = False
    try:
        y.backward()
    except Exception:
        threw = True
    check("backward on a no_grad result raises", threw)

    # and training still works after a no_grad block
    for t in m.parameters():
        t.reset_grad()
    out = m.forward(x)
    p.mean(p.multiply(out, out), -1).backward()
    check("training is unaffected afterwards",
          all(any(v != 0.0 for v in t.grad) for t in m.parameters()))


def main():
    print("Python API checks")
    test_bindings_present()
    test_reproducibility()
    test_gradcheck_from_python()
    test_python_gpt_matches_cpp()
    test_lazy_grad_and_no_grad()
    test_save_load()

    print("\n%d checks, %d failed" % (CHECKS[0], len(FAILURES)))
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
