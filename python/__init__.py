"""Prometheus: a deep learning library written from scratch in C++.

The compiled core is prometheus._C; everything it exports is re-exported
here, so `import prometheus as p; p.Tensor(...)` reads as it always has.
The pure-Python parts live alongside it:

    prometheus.models     MLP, CNN, ResNet, UNet, VAE, GAN, and the visualiser
    prometheus.gpt        a GPT written in Python on top of the C++ ops
    prometheus.visualise  3-D architecture view
"""
from ._C import *  # noqa: F401,F403
from ._C import __doc__  # noqa: F401


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
