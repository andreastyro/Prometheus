"""Running an IR graph on GPU tensors: element-wise regions become generated
kernels, everything else runs as the library's ops.

Partition: every non-element-wise value (a reduction) is computed by its
runtime op; every graph output and every reduction argument that is
element-wise is computed by one fused kernel whose leaves are the values
already computed -- inputs, constants aside, and reduction results. Work
shared by two such kernels is recomputed in each (element-wise work is
cheap; the memory traffic of a materialised intermediate is not).

Each fused kernel is differentiable: its backward is generated from the
IR's reverse-mode derivative, and when the gradient is itself recorded the
derivative is built as fused kernels in turn (so to any order). Reductions
are differentiated by their runtime ops.
"""
from .. import _C
from . import codegen
from .ir import Graph

_cache = {}        # (graph identity, root) -> (fwd code, bwd code, mask, leaf ids)


def _region(graph, root):
    """The leaves of the element-wise region computing `root`: inputs and
    non-element-wise values it reads (constants are inlined)."""
    leaves, seen, stack = [], set(), [root]
    while stack:
        v = stack.pop()
        if v.id in seen:
            continue
        seen.add(v.id)
        if v.op == "const":
            continue
        if v.op == "input" or not v.elementwise:
            if v is not root:
                leaves.append(v)
                continue
        stack.extend(v.args)
    leaves.sort(key=lambda v: v.id)
    return leaves


def _compile(graph, root):
    key = (id(graph), root.id)
    if key in _cache:
        return _cache[key]
    leaves = _region(graph, root)
    names = {v.id: "x%d" % i for i, v in enumerate(leaves)}
    fwd = codegen.emit(graph, [(root, "out")], names)
    g = graph.make("input", (), {"index": -1, "seed": True})    # the upstream gradient, a leaf
    grads = graph.grad(root, leaves, seed=g)
    mask = [0 if Graph.is_const(d, 0.0) else 1 for d in grads]
    bnames = dict(names)
    bnames[g.id] = "g"
    bwd = codegen.emit(graph, [(d, "d%d" % j) for j, (d, m) in enumerate(zip(grads, mask)) if m], bnames)
    entry = (fwd, bwd, mask, leaves, grads, g)
    _cache[key] = entry
    return entry


def _run_region(graph, root, env):
    fwd, bwd, mask, leaves, grads, g = _compile(graph, root)
    ins = [env[v.id] for v in leaves]

    def higher(tensors, G):
        # the gradient recorded: each input's derivative g * d_j as a fused
        # region of its own over the same leaves plus g
        env2 = {v.id: t for v, t in zip(leaves, tensors)}
        env2[g.id] = G
        return [_run_region(graph, d, env2) if m else None for d, m in zip(grads, mask)]

    return _C._fused_map(fwd, bwd, mask, ins, higher)


# ── backend selection ─────────────────────────────────────────────────────────
#
# Which implementation runs each part of a graph, decided here, from the IR:
#   matmul                      -> cuBLAS (the library's matmul, TF32 / bf16 as set)
#   softmax, sum / mean         -> the library's kernels (row-wise, warp-per-row)
#   reshape, transpose          -> views: no kernel
#   an element-wise region      -> one generated kernel (codegen + NVRTC)
#     ... of a single op whose inputs all have the output's shape
#                               -> the library's own kernel (vectorised float4
#                                  loads, nothing to compile) instead
_SINGLE = {"add": "add", "sub": "subtract", "mul": "multiply", "div": "divide",
           "exp": "exp_op", "log": "log_op", "sqrt": "sqrt_op", "abs": "abs_op", "tanh": "tanh_op",
           "sigmoid": "sigmoid", "relu": "relu", "gelu": "gelu"}


def backend(graph, v):
    """The implementation chosen for value v (see the table above)."""
    if v.op == "matmul":
        return "cublas"
    if v.op in ("softmax", "sum", "mean"):
        return "library"
    if v.op in ("reshape", "transpose"):
        return "view"
    if v.elementwise:
        if (v.op in _SINGLE and all(a.op not in ("const",) and a.shape == v.shape for a in v.args)
                and all(not a.elementwise for a in v.args)):
            return "library-elementwise"
        return "generated"
    return "none"


def _run_library(v, env):
    a = [env[x.id] for x in v.args]
    if v.op == "matmul":
        return _C.matmul(a[0], a[1])
    if v.op == "softmax":
        return _C.softmax(a[0])
    if v.op == "transpose":
        return _C.transpose(a[0])
    if v.op == "reshape":
        return _C.reshape(a[0], list(v.attrs["shape"]))
    if v.op in ("sum", "mean"):
        ax = v.attrs["axis"] % len(a[0].shape)
        return (_C.sum if v.op == "sum" else _C.mean)(a[0], ax, v.attrs["keepdim"])
    if v.op in _SINGLE:
        return getattr(_C, _SINGLE[v.op])(*a)
    raise ValueError("executor: no library op for " + v.op)


# ── memory: lifetimes ─────────────────────────────────────────────────────────
#
# Every computed tensor is dropped from the executor's environment right after
# its last reader runs, so its block goes back to the caching allocator and
# the next intermediate of that size class reuses it within the same call
# (tensors autograd keeps for the backward stay alive through their nodes).

def _reads(graph, v):
    """The computed tensors v's implementation reads: a generated region its
    leaves (inputs, reduction / library results), anything else its arguments."""
    return _region(graph, v) if v.elementwise else list(v.args)


def _last_use(graph, order):
    last = {}
    for step, v in enumerate(order):
        for a in _reads(graph, v):
            last[a.id] = step
    for o in graph.outputs:
        last[o.id] = len(order)
    return last


def _materialised(graph):
    """The values a run computes as tensors, in order: library ops,
    reductions, views, and the element-wise values they or the outputs read
    (each the root of one generated region)."""
    live = graph.live()
    roots = set(o.id for o in graph.outputs if o.elementwise)
    for v in live:
        if not v.elementwise and v.op not in ("input", "const"):
            roots.update(a.id for a in v.args if a.elementwise)
    return [v for v in live if (not v.elementwise and v.op not in ("input", "const")) or v.id in roots]


def plan(graph, shapes):
    """The memory plan: (peak bytes held by intermediates with release at
    last use, the same without release), from the inferred shapes."""
    graph.infer(shapes)
    order = _materialised(graph)
    last = _last_use(graph, order)
    size = {v.id: 4 * _numel(v.shape) for v in order if v.op not in ("reshape", "transpose")}
    held, peak, total = set(), 0, 0
    for step, v in enumerate(order):
        held.add(v.id)
        total += size.get(v.id, 0)
        peak = max(peak, sum(size.get(i, 0) for i in held))
        held = {i for i in held if last.get(i, -1) > step}
    return peak, total


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def run(graph, tensors):
    """The graph's outputs on these tensors (one per input)."""
    graph.infer([tuple(t.shape) for t in tensors])
    env = {v.id: t for v, t in zip(graph.inputs, tensors)}
    order = _materialised(graph)
    last = _last_use(graph, order)
    for step, v in enumerate(order):
        b = backend(graph, v)
        if b == "generated" or (b == "library-elementwise" and any(a.id not in env for a in v.args)):
            env[v.id] = _run_region(graph, v, env)
        else:
            env[v.id] = _run_library(v, env)
        for a in _reads(graph, v):             # release what nothing later reads
            if last.get(a.id, -1) <= step and a.id in env and a.op != "input":
                del env[a.id]
    return [env[o.id] for o in graph.outputs]
