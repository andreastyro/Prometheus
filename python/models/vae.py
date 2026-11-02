"""VAE — encoder, reparameterisation, decoder."""

import prometheus as p

from .base import Model, activation


class VAE(Model):
    """Variational autoencoder over flat vectors.

        VAE(input_dim, latent_dim, hidden_dims=[128])

    forward(x) returns (reconstruction, mu, logvar). The caller needs all three
    because the loss has two terms:

        recon = mse_loss(reconstruction, x)          (or bce for [0,1] data)
        kl    = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
        loss  = recon + beta * kl

    kl_loss() below computes that second term from the ops, so it stays in the
    autograd graph.

    The reparameterisation draws eps ~ N(0, 1) and forms z = mu + exp(0.5 *
    logvar) * eps. eps is a constant with respect to the parameters, so it is
    created without requires_grad and the gradient flows through mu and logvar
    only -- which is the entire point of the trick.
    """

    def __init__(self, input_dim, latent_dim, hidden_dims=(128,),
                 activation_name="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims)

        # ---- encoder ----
        dims = [input_dim] + self.hidden_dims
        self.encoder = []
        for i in range(len(dims) - 1):
            lin = self.add("enc_linear%d" % (i + 1), p.Linear(dims[i], dims[i + 1]))
            act = self.add("enc_%s%d" % (activation_name, i + 1),
                           activation(activation_name))
            self.encoder += [lin, act]

        self.fc_mu = self.add("fc_mu", p.Linear(dims[-1], latent_dim))
        self.fc_logvar = self.add("fc_logvar", p.Linear(dims[-1], latent_dim))

        # ---- decoder ----
        rev = [latent_dim] + list(reversed(self.hidden_dims))
        self.decoder = []
        for i in range(len(rev) - 1):
            lin = self.add("dec_linear%d" % (i + 1), p.Linear(rev[i], rev[i + 1]))
            act = self.add("dec_%s%d" % (activation_name, i + 1),
                           activation(activation_name))
            self.decoder += [lin, act]

        self.fc_out = self.add("dec_output", p.Linear(rev[-1], input_dim))
        self.out_act = self.add("sigmoid", p.Sigmoid())

    # ------------------------------------------------------------------ pieces
    def encode(self, x):
        """x -> (mu, logvar)"""
        for layer in self.encoder:
            x = layer.forward(x)
        return self.fc_mu.forward(x), self.fc_logvar.forward(x)

    def reparameterise(self, mu, logvar):
        """z = mu + exp(0.5 * logvar) * eps,  eps ~ N(0, 1)"""
        std = p.exp_op(p.multiply(0.5, logvar))
        eps = p.Tensor.randn(mu.shape)        # constant: no requires_grad
        return p.add(mu, p.multiply(std, eps))

    def decode(self, z):
        for layer in self.decoder:
            z = layer.forward(z)
        return self.out_act.forward(self.fc_out.forward(z))

    # ----------------------------------------------------------------- forward
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

    def kl_loss(self, mu, logvar):
        """-0.5 * mean(1 + logvar - mu^2 - exp(logvar)), built from ops."""
        mu_sq = p.multiply(mu, mu)
        var = p.exp_op(logvar)
        inner = p.subtract(p.subtract(p.add(1.0, logvar), mu_sq), var)
        return p.multiply(-0.5, p.mean(inner, -1))
