"""UNet — encoder/decoder with concatenating skip connections."""

import prometheus as p

from .base import Model


class _ConvBlock:
    """Conv -> GroupNorm -> ReLU, twice. The repeated unit of a UNet stage."""

    def __init__(self, in_c, out_c, groups):
        self.conv1 = p.Conv2D(in_c, out_c, 3, 1, 1)
        self.norm1 = p.GroupNorm(groups, out_c)
        self.conv2 = p.Conv2D(out_c, out_c, 3, 1, 1)
        self.norm2 = p.GroupNorm(groups, out_c)
        self.relu = p.ReLU()

    def forward(self, x):
        x = self.relu.forward(self.norm1.forward(self.conv1.forward(x)))
        return self.relu.forward(self.norm2.forward(self.conv2.forward(x)))

    def parameters(self):
        return (self.conv1.parameters() + self.norm1.parameters() +
                self.conv2.parameters() + self.norm2.parameters())


class UNet(Model):
    """Encoder-decoder with skip connections, the standard diffusion backbone.

        UNet(in_channels, num_classes, base_channels=16, depth=2)

    Encoder halves the spatial size and doubles the channels at each level.
    Decoder mirrors it with ConvTranspose2D, and at each level concatenates the
    matching encoder feature map along the channel axis before its conv block.
    That concatenation is why the library needed concat_op: joining two feature
    maps on the channel axis is not expressible from the elementwise ops.

    image_size must be divisible by 2**depth, otherwise the decoder would
    produce a different spatial size from its skip partner and the concat would
    be invalid. That is checked up front rather than failing mid-forward.

    Input  [batch, in_channels, size, size]
    Output [batch, num_classes, size, size]
    """

    def __init__(self, in_channels, num_classes, base_channels=16,
                 depth=2, image_size=32, groups=4):
        super().__init__()
        if image_size % (2 ** depth) != 0:
            raise ValueError(
                "image_size %d must be divisible by 2**depth (%d) so each "
                "decoder level matches its skip connection"
                % (image_size, 2 ** depth))

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth

        # ---- encoder ----
        self.encoders, self.pools = [], []
        prev = in_channels
        for d in range(depth):
            c = base_channels * (2 ** d)
            if c % groups != 0:
                raise ValueError("encoder channels %d not divisible by groups %d"
                                 % (c, groups))
            blk = _ConvBlock(prev, c, groups)
            self.encoders.append(blk)
            self.add("enc%d" % (d + 1), blk)
            pool = p.MaxPool2D(2)
            self.pools.append(pool)
            self.add("pool%d" % (d + 1), pool)
            prev = c

        # ---- bottleneck ----
        bottleneck_c = base_channels * (2 ** depth)
        self.bottleneck = _ConvBlock(prev, bottleneck_c, groups)
        self.add("bottleneck", self.bottleneck)
        prev = bottleneck_c

        # ---- decoder ----
        self.ups, self.decoders = [], []
        for d in reversed(range(depth)):
            skip_c = base_channels * (2 ** d)
            up = p.ConvTranspose2D(prev, skip_c, 2, 2)   # kernel 2, stride 2: doubles
            self.ups.append(up)
            self.add("up%d" % (d + 1), up)
            # after concat the channel count is skip_c (from up) + skip_c (skip)
            blk = _ConvBlock(skip_c * 2, skip_c, groups)
            self.decoders.append(blk)
            self.add("dec%d" % (d + 1), blk)
            prev = skip_c

        self.add("output", p.Conv2D(prev, num_classes, 1))
        self.head = self._layers[-1][1]

    def forward(self, x):
        skips = []
        for blk, pool in zip(self.encoders, self.pools):
            x = blk.forward(x)
            skips.append(x)              # save before downsampling
            x = pool.forward(x)

        x = self.bottleneck.forward(x)

        for up, blk, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up.forward(x)
            x = p.concat_op([x, skip], 1)   # join on the channel axis
            x = blk.forward(x)

        return self.head.forward(x)
