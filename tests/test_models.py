"""
Checks every Prometheus Model: output shape, parameter bookkeeping, and that
gradients actually reach the parameters.

The gradient check is the part that matters. A model is pure composition, so
its forward pass is easy to get right and easy to get subtly disconnected --
exactly the failure mode that left Flatten freezing every convolution and
Dropout freezing everything above it. Here every model is trained for one step
and every parameter is asserted to have received a non-zero gradient.

Run directly:  python tests/test_models.py
"""

import os
import sys

import prometheus as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prometheus.models import CNN, GAN, MLP, VAE, ResNet, UNet  # noqa: E402

FAILURES = []
CHECKS = [0]


def check(name, cond, detail=""):
    CHECKS[0] += 1
    if not cond:
        FAILURES.append(name + ((" (" + detail + ")") if detail else ""))
    print("  %-56s %s" % (name, "ok" if cond else "** FAIL **"))


def tensor(shape, grad=False):
    n = 1
    for d in shape:
        n *= d
    t = p.Tensor(shape)
    # spread across a range, offset so nothing lands exactly on a ReLU kink
    t.data = [(-0.8 + 1.6 * i / max(1, n - 1)) + 0.013 for i in range(n)]
    if grad:
        t.requires_grad = True
    return t


def all_params_get_gradient(name, model, loss):
    """Backward once, then assert every parameter received something."""
    params = model.parameters() if not isinstance(model, list) else model
    for t in params:
        t.reset_grad()
    loss.backward()

    dead = [i for i, t in enumerate(params)
            if all(v == 0.0 for v in t.grad)]
    check("%s: all %d parameters receive gradient" % (name, len(params)),
          not dead, "dead parameter indices: %s" % dead[:8])

    finite = all(all(v == v and abs(v) != float("inf") for v in t.grad)
                 for t in params)
    check("%s: gradients are finite" % name, finite)


def check_shape(name, got, want):
    check("%s: output shape %s" % (name, want), list(got) == list(want),
          "got %s" % got)


def no_duplicate_parameters(name, model):
    """parameters() must de-duplicate: a shared tensor updated twice diverges."""
    params = model.parameters()
    ids = [id(t) for t in params]
    check("%s: parameters() has no duplicates" % name,
          len(ids) == len(set(ids)),
          "%d entries, %d unique" % (len(ids), len(set(ids))))


def test_mlp():
    print("\n=== MLP ===")
    m = MLP(4, 3, hidden_dims=(8, 8))
    x = tensor([5, 4], grad=True)
    out = m.forward(x)
    check_shape("MLP", out.shape, [5, 3])
    no_duplicate_parameters("MLP", m)
    check("MLP: parameter count", m.num_parameters() == 4 * 8 + 8 + 8 * 8 + 8 + 8 * 3 + 3,
          str(m.num_parameters()))
    all_params_get_gradient("MLP", m, p.mean(p.multiply(out, out), -1))

    # dropout variant must still be fully connected
    md = MLP(4, 2, hidden_dims=(6,), dropout=0.3)
    o = md.forward(tensor([3, 4], grad=True))
    check_shape("MLP+dropout", o.shape, [3, 2])
    all_params_get_gradient("MLP+dropout", md, p.mean(p.multiply(o, o), -1))


def test_cnn():
    print("\n=== CNN ===")
    m = CNN(1, 10, channels=(8, 16), image_size=16, groups=4)
    x = tensor([2, 1, 16, 16], grad=True)
    out = m.forward(x)
    check_shape("CNN", out.shape, [2, 10])
    no_duplicate_parameters("CNN", m)
    all_params_get_gradient("CNN", m, p.mean(p.multiply(out, out), -1))

    # a bad configuration must be rejected, not silently mis-shaped
    threw = False
    try:
        CNN(1, 2, channels=(8, 8, 8, 8, 8, 8), image_size=8)
    except ValueError:
        threw = True
    check("CNN: too many pooling stages raises", threw)


def test_resnet():
    print("\n=== ResNet ===")
    m = ResNet(10, depth=18, in_channels=3, base_channels=8,
               image_size=32, groups=4)
    x = tensor([2, 3, 32, 32], grad=True)
    out = m.forward(x)
    check_shape("ResNet18", out.shape, [2, 10])
    no_duplicate_parameters("ResNet18", m)
    all_params_get_gradient("ResNet18", m, p.mean(p.multiply(out, out), -1))

    m34 = ResNet(4, depth=34, in_channels=1, base_channels=8,
                 image_size=32, groups=4)
    o = m34.forward(tensor([1, 1, 32, 32], grad=True))
    check_shape("ResNet34", o.shape, [1, 4])

    threw = False
    try:
        ResNet(10, depth=50)
    except ValueError:
        threw = True
    check("ResNet: depth 50 raises rather than building the wrong net", threw)


def test_unet():
    print("\n=== UNet ===")
    m = UNet(1, 2, base_channels=8, depth=2, image_size=16, groups=4)
    x = tensor([1, 1, 16, 16], grad=True)
    out = m.forward(x)
    # a UNet must return to its input resolution
    check_shape("UNet", out.shape, [1, 2, 16, 16])
    no_duplicate_parameters("UNet", m)
    all_params_get_gradient("UNet", m, p.mean(p.multiply(out, out), -1))

    threw = False
    try:
        UNet(1, 2, depth=3, image_size=20)   # 20 is not divisible by 8
    except ValueError:
        threw = True
    check("UNet: indivisible image_size raises", threw)


def test_vae():
    print("\n=== VAE ===")
    m = VAE(6, 2, hidden_dims=(8,))
    x = tensor([4, 6], grad=True)
    rec, mu, logvar = m.forward(x)
    check_shape("VAE reconstruction", rec.shape, [4, 6])
    check_shape("VAE mu", mu.shape, [4, 2])
    check_shape("VAE logvar", logvar.shape, [4, 2])
    no_duplicate_parameters("VAE", m)

    # Sigmoid output must be a valid probability. The bound is inclusive:
    # float32 sigmoid saturates to exactly 0.0 or 1.0 once |x| is large, so a
    # strict inequality is not guaranteed and would fail depending on init.
    check("VAE: reconstruction in [0, 1]",
          all(0.0 <= v <= 1.0 for v in rec.data),
          "min %.4g max %.4g" % (min(rec.data), max(rec.data)))

    recon = p.mse_loss(rec, x)
    kl = m.kl_loss(mu, logvar)
    all_params_get_gradient("VAE", m, p.add(recon, kl))

    # the KL term alone must reach fc_mu and fc_logvar, or the latent space
    # never regularises
    for t in m.parameters():
        t.reset_grad()
    rec2, mu2, lv2 = m.forward(x)
    m.kl_loss(mu2, lv2).backward()
    mu_grad = sum(abs(v) for v in m.fc_mu.weights.grad)
    lv_grad = sum(abs(v) for v in m.fc_logvar.weights.grad)
    check("VAE: KL term reaches fc_mu", mu_grad > 0)
    check("VAE: KL term reaches fc_logvar", lv_grad > 0)

    # reparameterisation must be stochastic, and reproducible under a seed
    p.Tensor.manual_seed(3)
    a = m.reparameterise(mu, logvar).data
    b = m.reparameterise(mu, logvar).data
    check("VAE: reparameterisation is stochastic", a != b)
    p.Tensor.manual_seed(3)
    c = m.reparameterise(mu, logvar).data
    check("VAE: reparameterisation is seed-reproducible", a == c)


def test_gan():
    print("\n=== GAN ===")
    g = GAN(8, img_size=16, img_channels=1, base_channels=8, groups=4)
    z = g.sample_latent(2)
    img = g.forward(z)
    check_shape("Generator output", img.shape, [2, 1, 16, 16])
    check("Generator: tanh output in (-1, 1)",
          all(-1.0 <= v <= 1.0 for v in img.data))

    score = g.discriminator.forward(img)
    check_shape("Discriminator output", score.shape, [2, 1])

    gp = g.generator_parameters()
    dp = g.discriminator_parameters()
    check("GAN: generator and discriminator parameters are disjoint",
          not (set(map(id, gp)) & set(map(id, dp))),
          "%d shared" % len(set(map(id, gp)) & set(map(id, dp))))
    check("GAN: combined count matches the two lists",
          len(g.parameters()) == len(gp) + len(dp))

    # generator loss must reach the generator and leave the discriminator alone
    for t in gp + dp:
        t.reset_grad()
    fake = g.forward(g.sample_latent(2))
    g_loss = p.mean(p.multiply(g.discriminator.forward(fake),
                               g.discriminator.forward(fake)), -1)
    g_loss.backward()
    gen_got = sum(1 for t in gp if any(v != 0.0 for v in t.grad))
    check("GAN: generator loss reaches the generator",
          gen_got == len(gp), "%d of %d" % (gen_got, len(gp)))

    # discriminator step on its own parameters
    for t in gp + dp:
        t.reset_grad()
    real = tensor([2, 1, 16, 16], grad=True)
    d_loss = p.mean(p.multiply(g.discriminator.forward(real),
                               g.discriminator.forward(real)), -1)
    d_loss.backward()
    disc_got = sum(1 for t in dp if any(v != 0.0 for v in t.grad))
    check("GAN: discriminator loss reaches the discriminator",
          disc_got == len(dp), "%d of %d" % (disc_got, len(dp)))


def test_summary_and_repr():
    print("\n=== introspection (what the visualiser will use) ===")
    m = CNN(1, 4, channels=(8,), image_size=8, groups=4)
    check("layers and layer_names line up",
          len(m.layers) == len(m.layer_names))
    check("summary includes a total row", "total" in m.summary())
    check("repr names the class", "CNN" in repr(m))
    check("every layer exposes parameters()",
          all(hasattr(l, "parameters") for l in m.layers))
    check("every layer exposes forward()",
          all(hasattr(l, "forward") for l in m.layers))


def test_norm_options():
    print("\n=== norm options ===")
    for kind in ("batch", "group", "none"):
        m = CNN(1, 4, channels=(8,), image_size=8, norm=kind, groups=4)
        out = m.forward(tensor([3, 1, 8, 8], grad=True))
        check_shape("CNN norm=%s" % kind, out.shape, [3, 4])
        all_params_get_gradient("CNN norm=%s" % kind, m,
                                p.mean(p.multiply(out, out), -1))

    # defaults follow the conventional choice per architecture
    check("CNN defaults to BatchNorm",
          type(CNN(1, 2, (8,), 8).layers[1]).__name__ == "BatchNorm")
    check("ResNet defaults to BatchNorm",
          type(ResNet(2, 18, 1, 8, 32).layers[1]).__name__ == "BatchNorm")
    check("UNet defaults to GroupNorm",
          type(UNet(1, 2, 8, 2, 16).encoders[0].norm1).__name__ == "GroupNorm")
    g = GAN(4, img_size=16, img_channels=1, base_channels=8)
    check("GAN discriminator defaults to GroupNorm",
          type(g.discriminator.layers[1]).__name__ == "GroupNorm")

    threw = False
    try:
        CNN(1, 2, (8,), 8, norm="bogus")
    except ValueError:
        threw = True
    check("an unknown norm name raises", threw)


def test_train_eval_cascade():
    print("\n=== train / eval cascade ===")
    m = CNN(1, 4, channels=(8, 8), image_size=16, norm="batch", groups=4)

    norms = [l for l in m.layers if type(l).__name__ == "BatchNorm"]
    check("CNN has BatchNorm layers to switch", len(norms) == 2)
    check("starts in training mode", m.training)

    m.eval()
    check("eval() reaches every BatchNorm",
          all(not n.training for n in norms))
    check("model reports not training", not m.training)

    m.train()
    check("train() restores every BatchNorm",
          all(n.training for n in norms))
    check("eval() returns self for chaining", m.eval() is m)

    # the bug the cascade prevents: BatchNorm in train mode normalises a single
    # sample by its own statistics, which makes every output identical
    m.train()
    x = tensor([1, 1, 16, 16])
    m.eval()
    eval_out = list(m.forward(x).data)
    m.train()
    train_out = list(m.forward(x).data)
    check("batch-of-one differs between train and eval mode",
          eval_out != train_out)

    # Dropout must be reached too, including inside an MLP
    md = MLP(4, 2, hidden_dims=(8,), dropout=0.5)
    drops = [l for l in md.layers if type(l).__name__ == "Dropout"]
    check("MLP exposes its Dropout", len(drops) == 1)
    md.eval()
    check("eval() reaches Dropout", not drops[0].training)
    xd = tensor([2, 4])
    a = list(md.forward(xd).data)
    b = list(md.forward(xd).data)
    check("Dropout is deterministic once in eval mode", a == b)
    md.train()
    p.Tensor.manual_seed(1)
    c = list(md.forward(xd).data)
    p.Tensor.manual_seed(2)
    d = list(md.forward(xd).data)
    check("Dropout is stochastic again after train()", c != d)

    # the cascade must recurse into nested holders and nested Models
    u = UNet(1, 2, base_channels=8, depth=2, image_size=16,
             norm="batch", groups=4)
    u.eval()
    inner = [u.encoders[0].norm1, u.bottleneck.norm2, u.decoders[0].norm1]
    check("cascade reaches norms inside UNet conv blocks",
          all(not n.training for n in inner))

    gan = GAN(4, img_size=16, img_channels=1, base_channels=8, norm="batch")
    gan.eval()
    gnorms = [l for l in gan.layers if type(l).__name__ == "BatchNorm"]
    check("cascade reaches both halves of a GAN",
          len(gnorms) > 0 and all(not n.training for n in gnorms))


def test_batchnorm_conv_semantics():
    print("\n=== BatchNorm on conv input ===")
    # statistics must be per channel over batch and spatial positions, not the
    # 2D interpretation that silently normalised the wrong groups
    bn = p.BatchNorm(3)
    x = tensor([2, 3, 2, 2], grad=True)
    out = bn.forward(x)
    check_shape("BatchNorm 4D", out.shape, [2, 3, 2, 2])

    vals = list(out.data)
    # with gamma=1, beta=0 each channel should come out zero-mean
    for c in range(3):
        idx = [(n * 3 + c) * 4 + s for n in range(2) for s in range(4)]
        mean = sum(vals[i] for i in idx) / len(idx)
        check("channel %d is zero-mean after BatchNorm" % c, abs(mean) < 1e-4,
              "mean %.3e" % mean)

    r = p.gradcheck(lambda: bn.forward(x), bn.parameters() + [x])
    check("BatchNorm 4D gradients", r["passed"], r["report"][:100])

    threw = False
    try:
        bn.forward(tensor([5]))
    except Exception:
        threw = True
    check("a 1D input raises rather than reading past the shape", threw)


def main():
    p.Tensor.manual_seed(1234)
    print("Prometheus Models")
    test_mlp()
    test_cnn()
    test_resnet()
    test_unet()
    test_vae()
    test_gan()
    test_norm_options()
    test_train_eval_cascade()
    test_batchnorm_conv_semantics()
    test_summary_and_repr()

    print("\n%d checks, %d failed" % (CHECKS[0], len(FAILURES)))
    if FAILURES:
        print("\nFailures:")
        for f in FAILURES:
            print("  -", f)
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
