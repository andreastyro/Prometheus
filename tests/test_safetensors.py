"""Model files: safetensors in both directions, every dtype, and the old format.

The reader and writer are this library's own, so the reference is the
official `safetensors` package: files written here must load there with
identical values, and files written there must load here. Without the package
the interop checks are skipped, not failed.
"""
import os
import struct
import sys
import tempfile

import numpy as np
import prometheus as p

try:
    import safetensors.numpy as stnp
except ImportError:
    stnp = None

CHECKS, FAILURES = [0], []
TMP = tempfile.mkdtemp(prefix="prom_st_")


def check(name, ok, detail=""):
    CHECKS[0] += 1
    if not ok:
        FAILURES.append(name)
    print("  %-56s %s %s" % (name, "ok" if ok else "** FAIL **", detail))


def path(name):
    return os.path.join(TMP, name)


def raises(fn, *needles):
    try:
        fn()
    except Exception as err:
        return all(n in str(err) for n in needles)
    return False


DTYPES = {  # prometheus dtype -> numpy dtype the values compare in
    "float32": np.float32, "float64": np.float64, "float16": np.float16,
    "int64": np.int64, "int32": np.int32, "bool": np.bool_,
}


def sample(dt, shape, rng):
    if dt == "bool":
        return rng.integers(0, 2, shape).astype(bool)
    if dt in ("int64", "int32"):
        big = 2 ** 40 + 7 if dt == "int64" else 2 ** 30 + 7   # past float32's exact range
        return rng.integers(-big, big, shape)
    return rng.uniform(-3, 3, shape)


def test_roundtrip():
    print("\n=== round trip through this library ===\n")
    rng = np.random.default_rng(0)
    tensors, expect = {}, {}
    for i, dt in enumerate(list(DTYPES) + ["bfloat16"]):
        vals = sample(dt, (3, 5), rng)
        t = p.Tensor([3, 5], [float(v) for v in vals.reshape(-1)], getattr(p, dt))
        tensors["t_" + dt] = t
        expect["t_" + dt] = np.array(t.numpy())
    # a non-contiguous view is written as its values, not its storage
    base = p.Tensor([4, 6], [float(v) for v in range(24)])
    tensors["view"] = p.transpose(base)
    expect["view"] = np.arange(24, dtype=np.float32).reshape(4, 6).T

    p.save_safetensors(path("rt.safetensors"), tensors, {"note": "x"})
    back, meta = p.load_safetensors(path("rt.safetensors"))
    check("names come back in the order written", list(back) == list(tensors))
    check("metadata comes back", meta.get("note") == "x")
    for name, want in expect.items():
        got = back[name]
        ok = (got.dtype == tensors[name].dtype and list(got.shape) == list(want.shape)
              and np.array_equal(np.array(got.numpy()), want))
        check("exact: " + name, ok)
    # int64 above 2^24 survives exactly -- it would not through float32
    big = np.array(back["t_int64"].numpy())
    check("int64 values above 2^40 exact", np.array_equal(big, expect["t_int64"]))


def test_interop():
    print("\n=== interop with the official safetensors package ===\n")
    if stnp is None:
        print("  (safetensors not installed: skipped)")
        return
    rng = np.random.default_rng(1)
    ours = {}
    for dt in DTYPES:
        vals = sample(dt, (2, 7), rng)
        ours[dt] = p.Tensor([2, 7], [float(v) for v in vals.reshape(-1)], getattr(p, dt))
    p.save_safetensors(path("ours.safetensors"), ours)
    theirs = stnp.load_file(path("ours.safetensors"))
    for dt, t in ours.items():
        a = theirs[dt]
        check("ours -> safetensors: " + dt,
              a.dtype == DTYPES[dt] and np.array_equal(a, np.array(t.numpy()).astype(DTYPES[dt])))

    arrays = {dt: sample(dt, (3, 4), rng).astype(np_dt) for dt, np_dt in DTYPES.items()}
    arrays["big"] = rng.uniform(-1, 1, (7, 300)).astype(np.float32)
    stnp.save_file(arrays, path("theirs.safetensors"), metadata={"source": "numpy"})
    back, meta = p.load_safetensors(path("theirs.safetensors"))
    for name, a in arrays.items():
        got = back[name]
        check("safetensors -> ours: " + name,
              list(got.shape) == list(a.shape) and np.array_equal(np.array(got.numpy()), a))
    check("their metadata reads", meta.get("source") == "numpy")


def test_models_and_checkpoints():
    print("\n=== save_model / checkpoints ===\n")
    p.Tensor.manual_seed(2)
    model = p.Sequential([p.Linear(5, 4), p.Tanh(), p.Linear(4, 2)])
    params = model.parameters()
    p.save_model(path("m.safetensors"), params)
    loaded = p.load_model(path("m.safetensors"))
    check("save_model / load_model round trip",
          len(loaded) == len(params) and all(np.array_equal(np.array(a.data), np.array(b.data))
                                             for a, b in zip(loaded, params)))
    if stnp is not None:
        check("a saved model opens in the safetensors package",
              len(stnp.load_file(path("m.safetensors"))) == len(params))

    p.save_checkpoint(path("c.safetensors"), params, 17, 0.123456789)
    ck, cparams = p.load_checkpoint(path("c.safetensors"))
    check("checkpoint epoch", ck.epoch == 17)
    check("checkpoint loss is the exact float", ck.loss == np.float32(0.123456789))
    check("checkpoint params", all(np.array_equal(np.array(a.data), np.array(b.data))
                                   for a, b in zip(cparams, params)))
    check("load_checkpoint on a plain model file explains itself",
          raises(lambda: p.load_checkpoint(path("m.safetensors")), "epoch"))


def test_legacy():
    print("\n=== files in the previous format still load ===\n")
    vals = [np.arange(6, dtype=np.float32).reshape(2, 3), np.array([1.5, -2.5], dtype=np.float32)]

    def block(arrs):
        out = struct.pack("<i", len(arrs))
        for a in arrs:
            out += struct.pack("<i", a.ndim) + b"".join(struct.pack("<i", d) for d in a.shape)
            out += a.astype("<f4").tobytes()
        return out

    with open(path("old_model.bin"), "wb") as f:
        f.write(block(vals))
    got = p.load_model(path("old_model.bin"))
    check("legacy model file", all(np.array_equal(np.array(g.data).reshape(v.shape), v)
                                   for g, v in zip(got, vals)))

    with open(path("old_ckpt.bin"), "wb") as f:
        f.write(b"CKPT" + struct.pack("<iif", 1, 9, 0.5) + block(vals))
    ck, got = p.load_checkpoint(path("old_ckpt.bin"))
    check("legacy checkpoint", ck.epoch == 9 and ck.loss == 0.5 and
          np.array_equal(np.array(got[1].data), vals[1]))


def test_malformed():
    print("\n=== malformed files are refused, not misread ===\n")
    good = path("ok.safetensors")
    p.save_safetensors(good, {"w": p.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])})
    raw = open(good, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = raw[8:8 + n]

    def variant(name, data):
        f = path(name)
        open(f, "wb").write(data)
        return f

    check("truncated data", raises(lambda: p.load_safetensors(variant("trunc", raw[:-3]))))
    bad_off = header.replace(b'"data_offsets":[0,16]', b'"data_offsets":[0,99]')
    check("byte range not matching the shape",
          raises(lambda: p.load_safetensors(variant("off", raw[:8] + bad_off + raw[8 + n:])),
                 "byte range"))
    u8 = header.replace(b'"F32"', b'"U8" ')
    check("an unsupported dtype is named",
          raises(lambda: p.load_safetensors(variant("u8", raw[:8] + u8 + raw[8 + n:])), "U8"))
    check("a header length past the end of the file",
          raises(lambda: p.load_safetensors(variant("len", struct.pack("<Q", 10 ** 9) + raw[8:]))))


def main():
    print("Model files: safetensors")
    test_roundtrip()
    test_interop()
    test_models_and_checkpoints()
    test_legacy()
    test_malformed()
    print("\n%d checks, %d failed" % (CHECKS[0], len(FAILURES)))
    print("PASS" if not FAILURES else "FAIL")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
