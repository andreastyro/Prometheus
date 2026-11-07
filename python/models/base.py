"""
Shared contract for every Prometheus Model.

A model is a callable object with the same two-method interface the library's
C++ Modules use:

    forward(x) -> Tensor
    parameters() -> list[Tensor]

Models live in Python rather than C++ on purpose. They are pure composition of
existing layers, so they need no new kernels, and keeping them here means an
architecture change costs no recompile. It also keeps them introspectable,
which is what the planned visualiser needs: walking layers, reading shapes and
pulling intermediate activations is free from Python and requires a new binding
for every member in C++.
"""

import contextlib

import prometheus as p


@contextlib.contextmanager
def no_grad():
    """Disable graph recording for the duration of the block.

    At inference there is nothing to backpropagate, so recording is pure
    overhead: each op would allocate a graph node holding its inputs and a
    closure, and the whole graph would stay alive until the result was dropped.

        with no_grad():
            logits = model.forward(x)
    """
    previous = p.is_grad_enabled()
    p.set_grad_enabled(False)
    try:
        yield
    finally:
        p.set_grad_enabled(previous)


class Model:
    """Base class. Subclasses set self._layers and may override forward."""

    def __init__(self):
        # ordered (name, layer) pairs; name is what the visualiser will label
        self._layers = []
        # losses collected by record_loss(), drawn by the dashboard
        self._loss_history = []

    # ------------------------------------------------------------------ layers
    def add(self, name, layer):
        """Register a layer in forward order. Returns the layer for chaining."""
        self._layers.append((name, layer))
        return layer

    @property
    def layers(self):
        """The registered layers, in forward order."""
        return [layer for _, layer in self._layers]

    @property
    def layer_names(self):
        return [name for name, _ in self._layers]

    # ----------------------------------------------------------------- forward
    def forward(self, x):
        """Run every registered layer in order. Override for non-linear graphs."""
        for _, layer in self._layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    # -------------------------------------------------------------- parameters
    def parameters(self):
        """Every learnable tensor, de-duplicated and in registration order.

        De-duplication matters for weight tying: a shared tensor must appear
        once, or an optimizer would apply its update twice.
        """
        seen, out = set(), []
        for _, layer in self._layers:
            for t in layer.parameters():
                key = id(t)
                if key not in seen:
                    seen.add(key)
                    out.append(t)
        return out

    def num_parameters(self):
        return sum(t.num_el() for t in self.parameters())

    # ----------------------------------------------------------------- trace
    # The visualiser needs the intermediate activations, not just the output.
    # The layers are pybind11 objects, so their forward cannot be wrapped from
    # Python; each model therefore declares how to walk itself. The default
    # covers any model that just runs its layers in order.
    def trace(self, x):
        """Run forward, returning [(layer_name, output_tensor), ...]."""
        out = []
        for name, layer in self._layers:
            x = layer.forward(x)
            out.append((name, x))
        return out

    # ------------------------------------------------------- train / eval mode
    # BatchNorm and Dropout behave differently during training and inference,
    # and a user should not have to know where they are. PyTorch solves this by
    # having Module.train(mode) recurse over its children; these do the same.
    # Without the cascade, forgetting to switch one BatchNorm gives predictions
    # that are wrong but still plausible, which is the worst kind of bug.
    def train(self, mode=True):
        """Put every layer into training mode. Returns self for chaining."""
        self._set_mode(mode)
        return self

    def eval(self):
        """Put every layer into inference mode. Returns self for chaining."""
        self._set_mode(False)
        return self

    @property
    def training(self):
        """True if no layer has been switched to inference mode."""
        return all(getattr(l, "training", True) for l in self._iter_mode_layers())

    def _iter_mode_layers(self):
        """Every layer that carries a training flag, recursing into holders."""
        for _, layer in self._layers:
            if hasattr(layer, "_layers"):          # a nested Model
                yield from layer._iter_mode_layers()
            elif hasattr(layer, "sublayers"):      # a plain composite holder
                for sub in layer.sublayers():
                    if hasattr(sub, "training"):
                        yield sub
            elif hasattr(layer, "training"):
                yield layer

    def _set_mode(self, mode):
        for layer in self._iter_mode_layers():
            # prefer the explicit methods, fall back to the flag
            if mode and hasattr(layer, "train"):
                layer.train()
            elif not mode and hasattr(layer, "eval"):
                layer.eval()
            else:
                layer.training = mode

    # ------------------------------------------------------------- visualiser
    def visualise(self, x=None, history=None, filename=None,
                  open_browser=True, **kwargs):
        """Write a self-contained HTML dashboard and open it in a browser.

        x        a sample batch; unlocks activations, feature maps, shape
                 annotations and the model-specific panel
        history  per-epoch losses for the loss curve, or omit it to use
                 whatever record_loss() has collected
        kwargs   real=, labels=, token_ids=, attention_layer=, live_seconds=

        Returns the path written.
        """
        from .visualiser import visualise as _vis
        if history is None and self._loss_history:
            history = self._loss_history
        return _vis(self, x=x, history=history, filename=filename,
                    open_browser=open_browser, **kwargs)

    def record_loss(self, value):
        """Append one epoch's loss, for the dashboard curve.

        Accepts a float or a scalar Tensor, so it drops straight into a
        training loop:  model.record_loss(loss)
        """
        if hasattr(value, "data"):
            value = value.data[0]
        self._loss_history.append(float(value))
        return value

    @property
    def loss_history(self):
        return list(self._loss_history)

    # ------------------------------------------------------------------ report
    def summary(self):
        """Layer names, parameter counts and the total."""
        lines = ["%-28s %12s" % ("layer", "params"), "-" * 41]
        for name, layer in self._layers:
            n = sum(t.num_el() for t in layer.parameters())
            lines.append("%-28s %12d" % (name, n))
        lines.append("-" * 41)
        lines.append("%-28s %12d" % ("total", self.num_parameters()))
        return "\n".join(lines)

    def __repr__(self):
        return "%s(%d layers, %d params)" % (
            type(self).__name__, len(self._layers), self.num_parameters())


def activation(name):
    """Map an activation name to a fresh layer instance."""
    table = {
        "relu":    p.ReLU,
        "sigmoid": p.Sigmoid,
        "tanh":    p.Tanh,
        "softmax": p.Softmax,
    }
    key = name.lower()
    if key not in table:
        raise ValueError("unknown activation %r; expected one of %s"
                         % (name, ", ".join(sorted(table))))
    return table[key]()

def norm_layer(kind, channels, groups=4):
    """Build a normalisation layer by name.

    "batch" — BatchNorm over the batch (and spatial positions, for conv input).
              The conventional choice for CNNs and ResNets, and what their
              published results assume. Requires eval() before inference.
    "group"  — GroupNorm within each sample. Independent of batch size, so a
              batch of one behaves like a batch of many. The usual choice for
              diffusion UNets and for GAN discriminators, where sharing
              statistics across a batch leaks information between samples.
    "none"   — identity, for ablations.
    """
    kind = (kind or "none").lower()
    if kind == "batch":
        return p.BatchNorm(channels)
    if kind == "group":
        if channels % groups != 0:
            raise ValueError("channels %d must be divisible by groups %d for "
                             "GroupNorm; pass a different groups=" % (channels, groups))
        return p.GroupNorm(groups, channels)
    if kind == "none":
        return Identity()
    raise ValueError('norm must be "batch", "group" or "none", got %r' % kind)


class Identity:
    """No-op layer, so norm="none" needs no branching at the call sites."""

    def forward(self, x):
        return x

    def parameters(self):
        return []
