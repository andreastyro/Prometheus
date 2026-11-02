"""GAN — a Generator and Discriminator pair."""

import prometheus as p

from .base import Model, norm_layer


class Generator(Model):
    """Maps a latent vector to an image.

    Starts with a Linear projection to a small spatial grid, then doubles the
    size with ConvTranspose2D(kernel 4, stride 2, pad 1) until it reaches
    img_size. Tanh output, so samples live in [-1, 1] and real data should be
    scaled to match.

    Input  [batch, latent_dim]
    Output [batch, img_channels, img_size, img_size]
    """

    def __init__(self, latent_dim, img_size=32, img_channels=1,
                 base_channels=32, norm="group", groups=4):
        super().__init__()
        if img_size % 4 != 0:
            raise ValueError("img_size must be divisible by 4, got %d" % img_size)

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.img_channels = img_channels

        # how many doublings to get from the start grid to img_size
        start = img_size
        n_up = 0
        while start % 2 == 0 and start > 4:
            start //= 2
            n_up += 1
        self.start_size = start
        self.n_up = n_up

        chans = base_channels * (2 ** n_up)
        if chans % groups != 0:
            raise ValueError("projection channels %d not divisible by groups %d"
                             % (chans, groups))
        self.proj_channels = chans

        self.project = self.add("project",
                                p.Linear(latent_dim, chans * start * start))
        self.add("project_relu", p.ReLU())

        self.ups = []
        prev = chans
        for i in range(n_up):
            out_c = max(base_channels, prev // 2)
            if out_c % groups != 0:
                out_c = groups * max(1, out_c // groups)
            up = self.add("up%d" % (i + 1),
                          p.ConvTranspose2D(prev, out_c, 4, 2, 1))
            nl = self.add("up_norm%d" % (i + 1), norm_layer(norm, out_c, groups))
            relu = self.add("up_relu%d" % (i + 1), p.ReLU())
            self.ups.append((up, nl, relu))
            prev = out_c

        self.to_image = self.add("to_image", p.Conv2D(prev, img_channels, 3, 1, 1))
        self.out_act = self.add("tanh", p.Tanh())

    def forward(self, z):
        x = self.project.forward(z)
        x = p.relu(x)
        batch = z.shape[0]
        x = p.reshape_op(x, [batch, self.proj_channels,
                             self.start_size, self.start_size])
        for up, norm, relu in self.ups:
            x = relu.forward(norm.forward(up.forward(x)))
        return self.out_act.forward(self.to_image.forward(x))


class Discriminator(Model):
    """Scores an image as real or generated.

    norm defaults to "group": BatchNorm in a discriminator shares statistics
    across the samples of a batch, which leaks information between real and
    generated examples and is a known way to get a GAN that never converges.

    Conv stride-2 stages halve the spatial size, then Flatten and a Linear to a
    single logit. No output activation: pair it with bce_loss after a Sigmoid,
    or use the logit directly.

    Input  [batch, img_channels, img_size, img_size]
    Output [batch, 1]
    """

    def __init__(self, img_size=32, img_channels=1, base_channels=32,
                 norm="group", groups=4):
        super().__init__()
        self.img_size = img_size

        size = img_size
        prev = img_channels
        i = 0
        while size > 4:
            out_c = base_channels * (2 ** i)
            self.add("conv%d" % (i + 1), p.Conv2D(prev, out_c, 4, 2, 1))
            self.add("norm%d" % (i + 1), norm_layer(norm, out_c, groups))
            self.add("relu%d" % (i + 1), p.ReLU())
            prev = out_c
            size //= 2
            i += 1

        self.feature_size = prev * size * size
        self.add("flatten", p.Flatten())
        self.add("output", p.Linear(self.feature_size, 1))


class GAN(Model):
    """Holds a Generator and Discriminator together.

        GAN(latent_dim, img_size=32)

    There is no single forward pass for a GAN: the two networks are trained
    against each other with separate optimizers. So forward(z) generates, and
    the two parameter lists are kept separate -- handing one optimizer the
    union of both would let the discriminator update on the generator loss,
    which is the classic way to get a GAN that never converges.
    """

    def __init__(self, latent_dim, img_size=32, img_channels=1,
                 base_channels=32, norm="group", groups=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, img_size, img_channels,
                                   base_channels, norm, groups)
        self.discriminator = Discriminator(img_size, img_channels,
                                           base_channels, norm, groups)
        self._layers = ([("generator." + n, l)
                         for n, l in self.generator._layers] +
                        [("discriminator." + n, l)
                         for n, l in self.discriminator._layers])

    def forward(self, z):
        """Generate images from latent vectors."""
        return self.generator.forward(z)

    def sample_latent(self, batch):
        return p.Tensor.randn([batch, self.latent_dim])

    def generator_parameters(self):
        return self.generator.parameters()

    def discriminator_parameters(self):
        return self.discriminator.parameters()
