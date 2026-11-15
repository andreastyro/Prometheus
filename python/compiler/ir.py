"""Prometheus compiler IR: a graph of typed values.

A Graph holds Values in creation order (always a topological order: a value's
arguments are created before it). Each Value has an op, its arguments,
attributes, and -- after infer() -- a shape and a dtype. Element-wise ops
broadcast like the tensor ops; reductions name an axis.

    g = Graph()
    x, t = g.input(), g.input()
    y = g.mul(x, g.unary("sigmoid", x))
    g.outputs = [g.sub(y, t)]
    g.infer([(4, 8), (8,)], ["float32"] * 2)

Passes: simplify (constant folding, algebraic identities, common
subexpressions -- every node is made through `make`, which merges an
identical (op, args, attrs) with the existing value), dead-code removal, and
reverse-mode differentiation of element-wise regions (`grad`). Consumers:
p.fuse (codegen.py, executor.py); later the whole-step compiler.
"""

ELEMENTWISE_UNARY = ("neg", "exp", "log", "sqrt", "abs", "tanh", "sigmoid", "relu", "gelu",
                     "sign", "step", "gelu_d")
ELEMENTWISE_BINARY = ("add", "sub", "mul", "div", "max", "min", "ge", "le")
REDUCTIONS = ("sum", "mean")


class Value:
    __slots__ = ("id", "op", "args", "attrs", "shape", "dtype", "graph")

    def __init__(self, graph, id_, op, args, attrs):
        self.graph, self.id, self.op, self.args, self.attrs = graph, id_, op, tuple(args), attrs
        self.shape, self.dtype = None, None

    @property
    def elementwise(self):
        return self.op in ELEMENTWISE_UNARY or self.op in ELEMENTWISE_BINARY or self.op == "pow"

    def __repr__(self):
        a = ", ".join("%%%d" % v.id for v in self.args)
        at = "" if not self.attrs else " " + repr(self.attrs)
        sh = "" if self.shape is None else " : %s %s" % (self.dtype, self.shape)
        return "%%%d = %s(%s)%s%s" % (self.id, self.op, a, at, sh)


def _broadcast(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        x = a[len(a) - 1 - i] if i < len(a) else 1
        y = b[len(b) - 1 - i] if i < len(b) else 1
        if x != y and x != 1 and y != 1:
            raise ValueError("shapes %s and %s do not broadcast" % (a, b))
        out.append(max(x, y))
    return tuple(reversed(out))


class Graph:
    def __init__(self):
        self.values = []
        self.inputs = []
        self.outputs = []
        self._memo = {}

    def __repr__(self):
        return "\n".join(["graph(%s):" % ", ".join("%%%d" % v.id for v in self.inputs)] +
                         ["  " + repr(v) for v in self.values] +
                         ["  return " + ", ".join("%%%d" % v.id for v in self.outputs)])

    # ── construction ───────────────────────────────────────────────────────
    def make(self, op, args=(), attrs=None):
        """A new value -- or the existing one with the same op, arguments and
        attributes (common subexpressions merge as they are built)."""
        key = (op, tuple(a.id for a in args), None if attrs is None else tuple(sorted(attrs.items())))
        if op != "input" and key in self._memo:
            return self._memo[key]
        v = Value(self, len(self.values), op, args, attrs or {})
        self.values.append(v)
        if op != "input":
            self._memo[key] = v
        return v

    def input(self):
        v = self.make("input", (), {"index": len(self.inputs)})
        self.inputs.append(v)
        return v

    def const(self, x):
        return self.make("const", (), {"value": float(x)})

    @staticmethod
    def is_const(v, x=None):
        return v.op == "const" and (x is None or v.attrs["value"] == x)

    # element-wise, with folding and identities
    def add(self, a, b):
        if self.is_const(a, 0.0): return b
        if self.is_const(b, 0.0): return a
        if self.is_const(a) and self.is_const(b): return self.const(a.attrs["value"] + b.attrs["value"])
        if b.id < a.id: a, b = b, a                      # commutative: one canonical order
        return self.make("add", (a, b))

    def sub(self, a, b):
        if self.is_const(b, 0.0): return a
        if self.is_const(a) and self.is_const(b): return self.const(a.attrs["value"] - b.attrs["value"])
        if a is b: return self.const(0.0)
        return self.make("sub", (a, b))

    def mul(self, a, b):
        if self.is_const(a, 0.0) or self.is_const(b, 0.0): return self.const(0.0)
        if self.is_const(a, 1.0): return b
        if self.is_const(b, 1.0): return a
        if self.is_const(a) and self.is_const(b): return self.const(a.attrs["value"] * b.attrs["value"])
        if b.id < a.id: a, b = b, a
        return self.make("mul", (a, b))

    def div(self, a, b):
        if self.is_const(a, 0.0): return self.const(0.0)
        if self.is_const(b, 1.0): return a
        if self.is_const(a) and self.is_const(b) and b.attrs["value"] != 0.0:
            return self.const(a.attrs["value"] / b.attrs["value"])
        return self.make("div", (a, b))

    def neg(self, a):
        if self.is_const(a): return self.const(-a.attrs["value"])
        return self.make("neg", (a,))

    def pow(self, a, e):
        e = float(e)
        if e == 1.0: return a
        if e == 0.0: return self.const(1.0)
        return self.make("pow", (a,), {"exponent": e})

    def unary(self, op, a):
        assert op in ELEMENTWISE_UNARY, op
        return self.make(op, (a,))

    def binary(self, op, a, b):
        assert op in ELEMENTWISE_BINARY, op
        return getattr(self, op)(a, b) if op in ("add", "sub", "mul", "div") else self.make(op, (a, b))

    def reduce(self, op, a, axis=-1, keepdim=True):
        assert op in REDUCTIONS, op
        return self.make(op, (a,), {"axis": int(axis), "keepdim": bool(keepdim)})

    # ── analysis ───────────────────────────────────────────────────────────
    def infer(self, input_shapes, input_dtypes=None):
        """Shapes and dtypes of every value, from the inputs'. Element-wise
        ops broadcast; reductions drop (or keep as 1) their axis. Values are
        computed in float32 (half inputs are read through a conversion)."""
        if len(input_shapes) != len(self.inputs):
            raise ValueError("graph has %d inputs, given %d shapes" % (len(self.inputs), len(input_shapes)))
        dts = input_dtypes or ["float32"] * len(self.inputs)
        for v, s, d in zip(self.inputs, input_shapes, dts):
            v.shape, v.dtype = tuple(s), d
        for v in self.live():                      # the outputs' cone (derivative nodes come later)
            if v.op == "input":
                continue
            if v.op == "const":
                v.shape, v.dtype = (), "float32"
            elif v.op in REDUCTIONS:
                s = v.args[0].shape
                ax = v.attrs["axis"] % len(s) if s else 0
                v.shape = tuple((1 if i == ax else n) for i, n in enumerate(s) if i != ax or v.attrs["keepdim"])
                v.dtype = "float32"
            else:
                shape = ()
                for a in v.args:
                    shape = _broadcast(shape, a.shape)
                v.shape, v.dtype = shape, "float32"
        return self

    def live(self, roots=None):
        """The values the roots (default: the outputs) depend on, in order."""
        need = set()
        stack = list(roots if roots is not None else self.outputs)
        while stack:
            v = stack.pop()
            if v.id in need:
                continue
            need.add(v.id)
            stack.extend(v.args)
        return [v for v in self.values if v.id in need]

    # ── differentiation (element-wise regions) ─────────────────────────────
    def grad(self, out, wrt, seed=None):
        """Reverse mode: d out / d w for each w in wrt, as values of this
        graph, each at out's (broadcast) shape -- reducing it to w's shape is
        the caller's (the runtime's sum_to). seed: the upstream gradient
        (default 1). Element-wise ops only: reductions are differentiated by
        the runtime ops they run as."""
        adj = {out.id: seed if seed is not None else self.const(1.0)}
        leaf = {w.id for w in wrt}
        for v in reversed(self.live([out])):
            if v.id not in adj or v.op in ("input", "const") or (v.id in leaf and v is not out):
                continue
            if not v.elementwise:
                raise ValueError("grad: %s is not element-wise" % v.op)
            g = adj[v.id]
            for a, d in zip(v.args, self._local(v)):
                if d is None:
                    continue
                c = self.mul(g, d)
                adj[a.id] = self.add(adj[a.id], c) if a.id in adj else c
        return [adj.get(w.id, self.const(0.0)) for w in wrt]

    def _local(self, v):
        """d v / d each argument."""
        op, a = v.op, v.args
        one = self.const(1.0)
        if op == "add": return [one, one]
        if op == "sub": return [one, self.const(-1.0)]
        if op == "mul": return [a[1], a[0]]
        if op == "div": return [self.div(one, a[1]), self.neg(self.div(v, a[1]))]
        if op == "neg": return [self.const(-1.0)]
        if op == "exp": return [v]
        if op == "log": return [self.div(one, a[0])]
        if op == "sqrt": return [self.div(self.const(0.5), v)]
        if op == "abs": return [self.unary("sign", a[0])]
        if op == "tanh": return [self.sub(one, self.mul(v, v))]
        if op == "sigmoid": return [self.mul(v, self.sub(one, v))]
        if op == "relu": return [self.unary("step", a[0])]
        if op == "gelu": return [self.unary("gelu_d", a[0])]
        if op == "pow":
            e = v.attrs["exponent"]
            return [self.mul(self.const(e), self.pow(a[0], e - 1.0))]
        if op in ("max", "min"):
            sel = self.make("ge" if op == "max" else "le", (a[0], a[1]))
            return [sel, self.sub(one, sel)]
        if op in ("ge", "le", "sign", "step"): return [None] * len(a)
        if op == "gelu_d":
            raise ValueError("grad: the second derivative of gelu is not implemented")
        raise ValueError("grad: no rule for " + op)
