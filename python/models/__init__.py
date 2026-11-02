"""
Prometheus Models — pre-built architectures with the Module contract.

Every model exposes forward() and parameters(), so it drops into the same
training loop as a hand-wired Sequential:

    from python.models import MLP

    model = MLP(input_dim=2, output_dim=1, hidden_dims=[16, 16])
    opt   = p.Adam(model.parameters(), lr=0.01)

    out  = model.forward(x)
    loss = p.mse_loss(out, y)
    loss.backward()
    opt.step()

These are written in Python rather than C++ because they are pure composition
of existing layers: no new kernels, no recompile to change an architecture, and
every layer stays reachable for the visualiser. The GPT in python/gpt.py follows
the same contract and is the reference implementation for the C++ p.GPT.
"""

from .base import Model
from .visualiser import build_dashboard, visualise
from .gan import GAN, Discriminator, Generator
from .mlp import CNN, MLP
from .resnet import ResNet
from .unet import UNet
from .vae import VAE

__all__ = [
    "Model",
    "visualise",
    "build_dashboard",
    "MLP",
    "CNN",
    "ResNet",
    "UNet",
    "VAE",
    "GAN",
    "Generator",
    "Discriminator",
]
