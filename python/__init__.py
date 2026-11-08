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
