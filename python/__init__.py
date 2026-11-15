"""Prometheus: a deep learning library written from scratch in C++.

The compiled core is prometheus._C; everything it exports is re-exported
here, so `import prometheus as p; p.Tensor(...)` reads as it always has.
The pure-Python parts live alongside it:

    prometheus.models     MLP, CNN, ResNet, UNet, VAE, GAN, and the visualiser
    prometheus.gpt        a GPT written in Python on top of the C++ ops
    prometheus.visualise  3-D architecture view
"""
import builtins  # the star import below shadows all, sum, ...

from ._C import *  # noqa: F401,F403
from ._C import __doc__  # noqa: F401
from ._C import _upload_async, _wait_upload  # noqa: F401  (not in the star import)


class autocast:
    """Mixed precision, like torch.autocast. Inside the block matmul and
    linear compute in `dtype` (float32 accumulation, the result
    rounded to dtype); softmax, log, exp, the reductions, norms and losses
    return float32. Pair float16 with GradScaler.

        with p.autocast(p.bfloat16):
            loss = model(x)
    """

    def __init__(self, dtype=None, enabled=True):
        from ._C import bfloat16, float32
        self.dtype = (dtype if dtype is not None else bfloat16) if enabled else float32
        self._previous = None

    def __enter__(self):
        from ._C import _set_autocast_dtype
        self._previous = _set_autocast_dtype(self.dtype)
        return self

    def __exit__(self, *exc):
        from ._C import _set_autocast_dtype
        _set_autocast_dtype(self._previous)
        return False


class _Capture:
    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        self.graph.capture_begin()
        return self.graph

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            # end the capture whatever happened, then let the error through
            try:
                self.graph.capture_end()
            except Exception:
                pass
            return False
        self.graph.capture_end()
        return False


def _graph_capture(self):
    """`with g.capture(): loss = step()` records the step's GPU work. Run the
    step a few times beforehand (kernels compiled, cuDNN plans made); inside,
    nothing may read a GPU value back (no .item(), no printing a GPU tensor,
    clip_grad_norm with return_norm=False)."""
    return _Capture(self)


CUDAGraph.capture = _graph_capture  # noqa: F405


def prefetch(batches, depth=2):
    """Batches of CPU tensors (an iterable of tensors or tuples / lists of
    them) as GPU tensors, the next `depth` already uploading while the
    current one is used: copies go through pinned memory on a second stream,
    and the compute stream waits for each on the GPU, never the CPU.

        for x, y in p.prefetch((x, y) for x, y in batches):
            loss = step(x, y)

    Without asynchronous uploads (no GPU, an old driver) batches are moved
    with .cuda() instead."""
    import collections

    def start(b):
        if isinstance(b, Tensor):  # noqa: F405
            return _upload_async(b)  # noqa: F405
        return type(b)(start(t) for t in b) if isinstance(b, (tuple, list)) else b

    def finish(u):
        if isinstance(u, tuple) and len(u) == 2 and isinstance(u[0], Tensor) and isinstance(u[1], int):  # noqa: F405
            _wait_upload(u[1])  # noqa: F405
            return u[0]
        return type(u)(finish(t) for t in u) if isinstance(u, (tuple, list)) else u

    def plain(b):
        if isinstance(b, Tensor):  # noqa: F405
            return b.cuda()
        return type(b)(plain(t) for t in b) if isinstance(b, (tuple, list)) else b

    if not cuda_uploads_available():  # noqa: F405
        for b in batches:
            yield plain(b)
        return
    it = iter(batches)
    queue = collections.deque()
    for b in it:
        queue.append(start(b))
        if len(queue) >= depth:
            break
    while queue:
        current = finish(queue.popleft())
        nxt = next(it, None)
        if nxt is not None:
            queue.append(start(nxt))
        yield current


class _Compiled:
    """What compile() returns: see compile."""

    def __init__(self, fn, warmup):
        self.fn, self.warmup = fn, warmup
        self._reset(None)
        self.eager_only = False

    def _reset(self, key):
        self.key, self.calls, self.graph, self.static, self.out = key, 0, None, None, None

    def __call__(self, *args):
        tensors = builtins.all(isinstance(a, Tensor) and a.device == cuda for a in args)  # noqa: F405
        if self.eager_only or not tensors or not cuda_graphs_available():  # noqa: F405
            return self.fn(*args)
        key = tuple((tuple(a.shape), a.dtype) for a in args)
        if key != self.key:                       # new shapes: warm up and capture again
            self._reset(key)
        if self.graph is None:
            if self.calls < self.warmup:
                self.calls += 1
                return self.fn(*args)
            # the captured inputs: copies the graph reads from on every replay
            self.static = [a.detach() for a in args]
            graph = CUDAGraph()  # noqa: F405
            try:
                with graph.capture():
                    self.out = self.fn(*self.static)
            except RuntimeError as e:
                if "graph capture" not in str(e):
                    raise
                # the step reads a value back (item(), printing, a returned
                # norm): it cannot be a graph; run it as it is from now on
                import warnings
                warnings.warn("prometheus.compile: running eagerly -- " + str(e).split("\n")[0])
                self.eager_only = True
                return self.fn(*args)
            self.graph = graph
        else:
            for s, a in zip(self.static, args):
                s.copy_from(a)
        self.graph.replay()
        return self.out


def compile(fn=None, *, warmup=3):
    """Stage one of a compiler, as torch.compile(mode="reduce-overhead"): the
    function (a training or inference step of GPU tensors) runs eagerly
    `warmup` times -- kernels compiled, cuDNN plans made -- then is captured
    in a CUDA Graph once and replayed: one launch for the whole step, the
    CPU's per-op work gone. Arguments are copied into the captured inputs
    on each call; the returned tensors are the captured outputs, refreshed in
    place. New shapes or dtypes capture again. A step that reads a GPU value
    back (.item(), printing, clip_grad_norm's returned norm) cannot be
    captured and keeps running eagerly, with a warning. Usable as a decorator:
    `@p.compile` or `@p.compile(warmup=5)`."""
    if fn is None:
        return lambda f: _Compiled(f, warmup)
    return _Compiled(fn, warmup)


from .fuse import fuse, fx  # noqa: E402,F401
