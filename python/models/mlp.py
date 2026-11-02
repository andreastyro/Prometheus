"""MLP and CNN — the two baseline classifiers."""

import prometheus as p

from .base import Model, activation, norm_layer


class MLP(Model):
    """Fully connected network.

        MLP(input_dim, output_dim, hidden_dims=[64, 64], activation="relu")

    Produces Linear -> act -> Linear -> act -> ... -> Linear. The final layer is
    left bare so the caller chooses the output nonlinearity: pair it with
    cross_entropy_sparse_seq for classification, or nothing for regression.

    Input  [batch, input_dim]
    Output [batch, output_dim]
    """

    def __init__(self, input_dim, output_dim, hidden_dims=(64, 64),
                 activation_name="relu", dropout=0.0):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)

        dims = [input_dim] + self.hidden_dims
        for i in range(len(dims) - 1):
            self.add("linear%d" % (i + 1), p.Linear(dims[i], dims[i + 1]))
            self.add("%s%d" % (activation_name, i + 1), activation(activation_name))
            if dropout > 0.0:
                self.add("dropout%d" % (i + 1), p.Dropout(dropout))

        self.add("output", p.Linear(dims[-1], output_dim))


class CNN(Model):
    """Convolutional classifier.

        CNN(in_channels, num_classes, channels=[16, 32], image_size=28)

    Each stage is Conv(3x3, pad 1) -> norm -> ReLU -> MaxPool(2), halving the
    spatial size, then a Flatten and a Linear head.

    norm defaults to "batch", which is what CNN architectures are tuned for.
    Call model.eval() before inference so BatchNorm uses its running estimate
    instead of the statistics of whatever batch it is handed. Pass norm="group"
    for batch-size independence, or norm="none".

    Input  [batch, in_channels, image_size, image_size]
    Output [batch, num_classes]
    """

    def __init__(self, in_channels, num_classes, channels=(16, 32),
                 image_size=28, norm="batch", groups=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = list(channels)
        self.image_size = image_size

        self.norm = norm

        size = image_size
        prev = in_channels
        for i, c in enumerate(self.channels):
            self.add("conv%d" % (i + 1), p.Conv2D(prev, c, 3, 1, 1))
            self.add("norm%d" % (i + 1), norm_layer(norm, c, groups))
            self.add("relu%d" % (i + 1), p.ReLU())
            self.add("pool%d" % (i + 1), p.MaxPool2D(2))
            prev = c
            size //= 2
            if size < 1:
                raise ValueError("image_size %d is too small for %d pooling stages"
                                 % (image_size, len(self.channels)))

        self.feature_size = prev * size * size
        self.add("flatten", p.Flatten())
        self.add("output", p.Linear(self.feature_size, num_classes))
