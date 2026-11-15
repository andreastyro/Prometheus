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


def _run_reduction(v, env):
    x = env[v.args[0].id]
    ax = v.attrs["axis"] % len(x.shape)
    fn = _C.sum if v.op == "sum" else _C.mean
    return fn(x, ax, v.attrs["keepdim"])


def run(graph, tensors):
    """The graph's outputs on these tensors (one per input)."""
    graph.infer([tuple(t.shape) for t in tensors])
    env = {v.id: t for v, t in zip(graph.inputs, tensors)}
    for v in graph.live():
        if v.id in env or v.op in ("input", "const") or v.elementwise:
            continue
        a = v.args[0]
        if a.id not in env:
            env[a.id] = _run_region(graph, a, env)
        env[v.id] = _run_reduction(v, env)
    outs = []
    for o in graph.outputs:
        if o.id not in env:
            env[o.id] = _run_region(graph, o, env) if o.elementwise or o.op == "const" else env[o.id]
        outs.append(env[o.id])
    return outs
