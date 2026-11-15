"""CUDA code for an element-wise region of the IR: one statement per value.

A region's leaves (inputs, and values computed outside it) are the kernel's
loads x0..x{k-1}; every other value is a `const float tN = ...;` in
topological order, so a value used twice is computed once (the derivative
graphs share most of their terms with the forward).
"""
from .ir import ELEMENTWISE_UNARY


def _num(v):
    if v in (float("inf"), float("-inf")):
        return ("-" if v < 0 else "") + "__int_as_float(0x7f800000)"
    if v != v:
        return "__int_as_float(0x7fc00000)"
    t = "%.9g" % v
    if not any(ch in t for ch in ".en"):
        t += ".0"
    return t + "f"


_UNARY_C = {
    "neg": "(-{0})", "exp": "expf({0})", "log": "logf({0})", "sqrt": "sqrtf({0})", "abs": "fabsf({0})",
    "tanh": "tanhf({0})", "sigmoid": "f_sigmoid({0})", "relu": "fmaxf({0}, 0.0f)", "gelu": "f_gelu({0})",
    "gelu_d": "f_gelu_d({0})", "step": "({0} > 0.0f ? 1.0f : 0.0f)",
    "sign": "({0} > 0.0f ? 1.0f : ({0} < 0.0f ? -1.0f : 0.0f))",
}
_BINARY_C = {
    "add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})", "div": "({0} / {1})",
    "max": "fmaxf({0}, {1})", "min": "fminf({0}, {1})",
    "ge": "({0} >= {1} ? 1.0f : 0.0f)", "le": "({0} <= {1} ? 1.0f : 0.0f)",
}


def emit(graph, targets, leaves):
    """Statements computing the targets from the leaves. leaves: value id ->
    variable name (x0, x1, ...); targets: (value, variable) pairs -- a value
    may be assigned to several variables. Returns the code."""
    roots = [v for v, _ in targets]
    name = dict(leaves)
    lines = []
    # the region: what the roots need, stopping at the leaves
    need, stack = set(), list(roots)
    while stack:
        v = stack.pop()
        if v.id in need:
            continue
        need.add(v.id)
        if v.id not in leaves:
            stack.extend(v.args)
    for v in (u for u in graph.values if u.id in need):
        if v.id in name:
            continue
        if v.op == "const":
            name[v.id] = _num(v.attrs["value"])
            continue
        if v.op == "input" or not v.elementwise:
            raise ValueError("codegen: %s is not a leaf of this region and not element-wise" % v.op)
        args = [name[a.id] for a in v.args]
        if v.op in _UNARY_C:
            expr = _UNARY_C[v.op].format(*args)
        elif v.op in _BINARY_C:
            expr = _BINARY_C[v.op].format(*args)
        elif v.op == "pow":
            e = v.attrs["exponent"]
            expr = "(%s * %s)" % (args[0], args[0]) if e == 2.0 else "powf(%s, %s)" % (args[0], _num(e))
        else:
            raise ValueError("codegen: no CUDA for " + v.op)
        name[v.id] = "t%d" % v.id
        lines.append("    const float t%d = %s;" % (v.id, expr))
    for r, var in targets:
        lines.append("    const float %s = %s;" % (var, name[r.id]))
    return "\n".join(lines)


assert set(_UNARY_C) == set(ELEMENTWISE_UNARY)
