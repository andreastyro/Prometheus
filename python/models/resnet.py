"""ResNet — stacks the library's ResidualBlock."""

import prometheus as p

from .base import Model

# depth -> blocks per stage, following the original paper for the
# basic-block variants
_STAGES = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
}


class ResNet(Model):
    """Residual classifier built from p.ResidualBlock.

        ResNet(num_classes, depth=18, in_channels=3, base_channels=16)

    Stem conv, then four stages. The first block of every stage after the first
    uses stride 2, halving the spatial size and doubling the channel count,
    which is where ResidualBlock inserts its projection shortcut. Ends with
    global average pooling and a Linear head.

    Depth 50 is not offered: it needs a bottleneck block (1x1 -> 3x3 -> 1x1 with
    a 4x channel expansion) which the library does not have yet. Asking for it
    raises rather than silently building a basic-block network of the wrong
    shape.

    Input  [batch, in_channels, size, size]
    Output [batch, num_classes]
    """

    def __init__(self, num_classes, depth=18, in_channels=3,
                 base_channels=16, image_size=32, groups=4):
        super().__init__()
        if depth not in _STAGES:
            if depth == 50:
                raise ValueError(
                    "depth 50 needs a bottleneck block (1x1-3x3-1x1 with 4x "
                    "expansion), which is not implemented; use 18 or 34")
            raise ValueError("depth must be one of %s, got %d"
                             % (sorted(_STAGES), depth))

        self.num_classes = num_classes
        self.depth = depth
        self.base_channels = base_channels

        blocks_per_stage = _STAGES[depth]

        # stem: keeps the spatial size, lifts channels to base_channels
        self.add("stem_conv", p.Conv2D(in_channels, base_channels, 3, 1, 1))
        self.add("stem_norm", p.GroupNorm(groups, base_channels))
        self.add("stem_relu", p.ReLU())

        size = image_size
        prev = base_channels
        for stage, n_blocks in enumerate(blocks_per_stage):
            out_c = base_channels * (2 ** stage)
            if out_c % groups != 0:
                raise ValueError("stage channel count %d not divisible by groups %d"
                                 % (out_c, groups))
            for b in range(n_blocks):
                # downsample on the first block of every stage but the first
                stride = 2 if (b == 0 and stage > 0) else 1
                self.add("stage%d_block%d" % (stage + 1, b + 1),
                         p.ResidualBlock(prev, out_c, stride, groups))
                prev = out_c
                if stride == 2:
                    size = (size + 1) // 2
            if size < 1:
                raise ValueError("image_size %d too small for depth %d"
                                 % (image_size, depth))

        # global average pool, implemented as an AvgPool over the whole map
        self.final_size = size
        self.add("global_pool", p.AvgPool2D(size))
        self.add("flatten", p.Flatten())
        self.add("output", p.Linear(prev, num_classes))
