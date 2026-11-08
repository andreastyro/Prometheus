"""Every parallelised layer gives the same answer on one thread as on many.

The gradchecks and numpy references use small shapes, which stay below the
thresholds where loops go parallel -- so on their own they never run the
parallel code. This runs each layer at a size that crosses them, forward and
backward, and compares the output and every gradient against the same run in
a child process limited to one thread. A data race, a scatter that two
threads write at once, a reduction that drops a partial: any of those shows up
here as a mismatch.

The two runs add in different orders, so they agree to rounding, not bit for
bit; each tensor's worst difference is measured against its own scale.
"""
import os
import subprocess
import sys

import numpy as np

SINGLE = "--single-thread" in sys.argv
if SINGLE:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

import prometheus as p  # noqa: E402  (after the thread limits are set)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_parallel_%s.npz")


def t(shape, seed, lo=-1.0, hi=1.0, grad=True):
    rng = np.random.default_rng(seed)
    x = p.Tensor(list(shape), [float(v) for v in rng.uniform(lo, hi, shape).reshape(-1)])
    if grad:
        x.requires_grad = True
        x.reset_grad()
    return x


def run_layer(name, layer, x):
    """Forward, a distinctly weighted loss, backward: output and all grads."""
    out = layer.forward(x)
    if isinstance(out, tuple):
        out = out[0]
    w = t(out.shape, 99, grad=False)
    p.sum(p.multiply(out, w), -1).backward()
    res = {name + ":out": np.array(out.data)}
    if x.requires_grad:
        res[name + ":dx"] = np.array(x.grad)
    for i, prm in enumerate(layer.parameters()):
        res["%s:dparam%d" % (name, i)] = np.array(prm.grad)
    return res


def compute():
    p.Tensor.manual_seed(5)
    r = {}
    r.update(run_layer("conv2d", p.Conv2D(16, 32, 3, 1, 1), t((16, 16, 24, 24), 1)))
    r.update(run_layer("convT2d", p.ConvTranspose2D(32, 16, 4, 2, 1), t((16, 32, 12, 12), 2)))
    r.update(run_layer("maxpool", p.MaxPool2D(2), t((16, 32, 24, 24), 3)))
    r.update(run_layer("maxpool_overlap", p.MaxPool2D(3, 1), t((16, 32, 24, 24), 4)))
    r.update(run_layer("avgpool", p.AvgPool2D(2), t((16, 32, 24, 24), 5)))
    r.update(run_layer("batchnorm", p.BatchNorm(32), t((16, 32, 24, 24), 6)))
    r.update(run_layer("groupnorm", p.GroupNorm(8, 32), t((16, 32, 24, 24), 7)))
    r.update(run_layer("layernorm", p.LayerNorm(256), t((512, 256), 8)))
    # The recurrent layers' default init is randn with std 1, which makes the
    # recurrence chaotic (spectral radius ~12 at H=128): any rounding
    # difference, such as BLAS summing in another order, grows ~10x per step
    # until two runs are unrelated -- even two numpy runs one 1e-7 nudge
    # apart. That would hide a race rather than find one, so they get
    # PyTorch's init range here, U(-1/sqrt(H), 1/sqrt(H)).
    for k, (name, layer) in enumerate((("rnn", p.RNN(64, 128)), ("lstm", p.LSTM(64, 128)),
                                       ("gru", p.GRU(64, 128)))):
        bound = 1.0 / np.sqrt(128)
        for j, prm in enumerate(layer.parameters()):
            prm.data = list(np.random.default_rng(100 + 10 * k + j).uniform(-bound, bound, prm.numel()))
        r.update(run_layer(name, layer, t((16, 32, 64), 9 + k)))

    emb = p.Embedding(1000, 256)
    rng = np.random.default_rng(12)
    ids = p.Tensor([32, 64], [float(v) for v in rng.integers(0, 50, 32 * 64)], p.int64)  # many repeats
    r.update(run_layer("embedding", emb, ids))

    big = t((300000,), 13)
    s = p.sum(big, -1)
    s.backward()
    r["sum:out"] = np.array(s.data)
    r["sum:dx"] = np.array(big.grad)

    for name, make in (("sgd", lambda w: p.SGD([w], 0.01, 0.9)),
                       ("adam", lambda w: p.Adam([w], 1e-3)),
                       ("adamw", lambda w: p.AdamW([w], 1e-3)),
                       ("rmsprop", lambda w: p.RMSprop([w], 1e-3))):
        w = t((200000,), 14)
        w.grad = list(np.random.default_rng(15).uniform(-1, 1, 200000))
        opt = make(w)
        for _ in range(3):
            opt.step()
        r[name + ":weights"] = np.array(w.data)
    return r


def main():
    if SINGLE:
        np.savez(OUT % "single", **compute())
        return

    many = compute()
    env = dict(os.environ)
    subprocess.run([sys.executable, os.path.abspath(__file__), "--single-thread"], check=True, env=env)
    single = dict(np.load(OUT % "single"))
    os.remove(OUT % "single")

    print("Parallel vs single-threaded, %d threads (backend %s)\n"
          % (p.backend()["threads"], p.backend()["matmul"]))
    failures = []
    for key in sorted(many):
        a, b = many[key], single[key]
        scale = max(float(np.max(np.abs(b))), 1e-12)
        err = float(np.max(np.abs(a - b))) / scale if a.size else 0.0
        ok = a.shape == b.shape and err <= 1e-4
        if not ok:
            failures.append(key)
        print("  %-28s rel. to scale %.1e  %s" % (key, err, "ok" if ok else "** FAIL **"))
    print("\n%d tensors compared, %d failed" % (len(many), len(failures)))
    print("PASS" if not failures else "FAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
