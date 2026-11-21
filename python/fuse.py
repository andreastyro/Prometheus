"""prometheus.fuse: a function of element-wise ops (and reductions along an
axis) as generated GPU kernels.

    @p.fuse
    def swish_mse(x, t):
        y = x * p.fx.sigmoid(x)
        return p.fx.mean((y - t) * (y - t))

The first call traces the function on symbols into the compiler's IR
(prometheus.compiler): arithmetic operators and the p.fx functions -- exp,
log, sqrt, abs, tanh, sigmoid, relu, gelu, pow, maximum, minimum, the
reductions sum and mean along an axis, and whole ops: `@` (matmul),
softmax, reshape, transpose, layernorm. A whole forward pass can be traced:
each element-wise region between library ops becomes one kernel forward and
one backward, generated from the IR's reverse-mode derivative (a gradient
that is itself recorded is built as fused kernels in turn, so to any order);
matmuls, softmax and reductions run as the library's ops (see
compiler/executor.py: backend selection, and intermediates released at their
last use). Inputs broadcast. On CPU tensors the same function runs as
ordinary ops. No Python control flow on tensor values.
"""
from . import _C
from .compiler import executor
from .compiler.ir import Graph


class Sym:
    """A traced value: what the function sees in place of a tensor."""
    __slots__ = ("g", "v")

    def __init__(self, g, v):
        self.g, self.v = g, v

    def _o(self, x):
        return x.v if isinstance(x, Sym) else self.g.const(x)

    def __add__(self, o): return Sym(self.g, self.g.add(self.v, self._o(o)))
    def __radd__(self, o): return Sym(self.g, self.g.add(self._o(o), self.v))
    def __sub__(self, o): return Sym(self.g, self.g.sub(self.v, self._o(o)))
    def __rsub__(self, o): return Sym(self.g, self.g.sub(self._o(o), self.v))
    def __mul__(self, o): return Sym(self.g, self.g.mul(self.v, self._o(o)))
    def __rmul__(self, o): return Sym(self.g, self.g.mul(self._o(o), self.v))
    def __truediv__(self, o): return Sym(self.g, self.g.div(self.v, self._o(o)))
    def __rtruediv__(self, o): return Sym(self.g, self.g.div(self._o(o), self.v))
    def __neg__(self): return Sym(self.g, self.g.neg(self.v))
    def __matmul__(self, o): return Sym(self.g, self.g.library("matmul", (self.v, self._o(o))))
    def __rmatmul__(self, o): return Sym(self.g, self.g.library("matmul", (self._o(o), self.v)))

    def __pow__(self, e):
        if not isinstance(e, (int, float)):
            raise TypeError("fuse: x ** y needs a number exponent")
        return Sym(self.g, self.g.pow(self.v, e))


def _unary(name, real):
    def f(x):
        return Sym(x.g, x.g.unary(name, x.v)) if isinstance(x, Sym) else real(x)
    f.__name__ = name
    return staticmethod(f)


def _binary(name, real):
    def f(a, b):
        s = a if isinstance(a, Sym) else b if isinstance(b, Sym) else None
        if s is None:
            return real(a, b)
        return Sym(s.g, s.g.make(name, (s._o(a), s._o(b))))
    f.__name__ = name
    return staticmethod(f)


def _reduction(name, real):
    def f(x, axis=-1, keepdim=True):
        if isinstance(x, Sym):
            return Sym(x.g, x.g.reduce(name, x.v, axis, keepdim))
        return real(x, axis % len(x.shape), keepdim)
    f.__name__ = name
    return staticmethod(f)


def _softmax(x):
    return Sym(x.g, x.g.library("softmax", (x.v,))) if isinstance(x, Sym) else _C.softmax(x)


def _reshape(x, shape):
    if isinstance(x, Sym):
        return Sym(x.g, x.g.library("reshape", (x.v,), shape=tuple(int(d) for d in shape)))
    return _C.reshape(x, list(shape))


def _transpose(x):
    return Sym(x.g, x.g.library("transpose", (x.v,))) if isinstance(x, Sym) else _C.transpose(x)


def _layernorm(x, gamma, beta, eps=1e-5):
    """LayerNorm over the last axis, written as reductions and element-wise
    ops: under p.fuse three generated kernels around two means."""
    m = fx.mean(x)
    d = x - m
    var = fx.mean(d * d)
    return d / fx.sqrt(var + eps) * gamma + beta


def _pow(x, e):
    return x ** e if isinstance(x, Sym) else _C.pow_op(x, float(e))


class fx:
    """Functions usable inside @p.fuse (on symbols) and outside it (on
    tensors, as the ordinary ops). Reductions default to the last axis,
    kept (keepdim=True), so they broadcast back."""
    exp = _unary("exp", _C.exp_op)
    log = _unary("log", _C.log_op)
    sqrt = _unary("sqrt", _C.sqrt_op)
    abs = _unary("abs", _C.abs_op)
    tanh = _unary("tanh", _C.tanh_op)
    sigmoid = _unary("sigmoid", _C.sigmoid)
    relu = _unary("relu", _C.relu)
    gelu = _unary("gelu", _C.gelu)
    pow = staticmethod(_pow)
    maximum = _binary("max", lambda a, b: _C.where(a >= b, a, b))
    minimum = _binary("min", lambda a, b: _C.where(a <= b, a, b))
    sum = _reduction("sum", _C.sum)
    mean = _reduction("mean", _C.mean)
    softmax = staticmethod(_softmax)
    reshape = staticmethod(_reshape)
    transpose = staticmethod(_transpose)
    layernorm = staticmethod(_layernorm)


class Fused:
    """What p.fuse returns. .fn is the plain function; .graph(k) its traced
    IR for k inputs."""

    def __init__(self, fn):
        self.fn = fn
        self.graphs = {}
        self.eager = False          # set when the function cannot be traced

    def graph(self, k):
        if k not in self.graphs:
            g = Graph()
            syms = [Sym(g, g.input()) for _ in range(k)]
            out = self.fn(*syms)
            g.outputs = [out.v if isinstance(out, Sym) else g.const(out)]
            self.graphs[k] = g
        return self.graphs[k]

    def __call__(self, *args):
        tensors = [a for a in args if isinstance(a, _C.Tensor)]
        if self.eager or not tensors or any(t.device != _C.cuda for t in tensors) or len(args) != len(tensors):
            return self.fn(*args)
        try:
            g = self.graph(len(args))
        except (TypeError, ValueError, AttributeError) as e:
            # an op the tracer does not know (a p.* function on a symbol, Python
            # control flow on a value): run the function as it is, from now on
            import warnings
            warnings.warn("prometheus.fuse: running %s unfused -- %s" % (getattr(self.fn, "__name__", "it"), e))
            self.eager = True
            return self.fn(*args)
        return executor.run(g, list(args))[0]


def fuse(fn):
    """See the module docstring: `@p.fuse` over an element-wise function."""
    return Fused(fn)
