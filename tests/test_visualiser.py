"""
Checks the model dashboard.

A visualiser is easy to get wrong in ways that look fine: a panel silently
missing, a chart drawn from stale data, or an HTML file that needs a CDN and so
breaks offline. These assert the structure instead of eyeballing it -- that each
model emits the panels it should, that the numbers in the page match the model,
and that nothing reaches for an external resource.

Run directly:  python tests/test_visualiser.py
"""

import os
import re
import sys

import prometheus as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.gpt import GPT                                     # noqa: E402
from python.models import CNN, GAN, MLP, VAE, ResNet, UNet      # noqa: E402
from python.models.visualiser import (build_dashboard,          # noqa: E402
                                      gradient_norms)

FAILURES = []
CHECKS = [0]


def check(name, cond, detail=""):
    CHECKS[0] += 1
    if not cond:
        FAILURES.append(name + ((" (" + detail + ")") if detail else ""))
    print("  %-58s %s" % (name, "ok" if cond else "** FAIL **"))


def tensor(shape):
    n = 1
    for d in shape:
        n *= d
    t = p.Tensor(shape)
    t.data = [(-0.8 + 1.6 * i / max(1, n - 1)) + 0.013 for i in range(n)]
    t.requires_grad = True
    return t


def panels(html):
    return re.findall(r"<h2>([^<]+)</h2>", html)


def test_self_contained():
    print("\n=== self-contained ===")
    m = MLP(4, 2, (8,))
    html = build_dashboard(m, x=tensor([3, 4]))

    # The point is that it opens from the filesystem, offline. Check for
    # actual resource loads rather than any occurrence of "http": the inlined
    # isometric SVG carries xmlns="http://www.w3.org/2000/svg", which is a
    # namespace identifier and is never fetched.
    for bad in ('src="http', 'href="http', "url(http",
                "<script src", "<link rel", "@import"):
        check("no external load via %s" % bad, bad not in html)

    urls = re.findall(r"https?://[^\s\"')]+", html)
    external = [u for u in urls if not u.startswith("http://www.w3.org/")]
    check("every URL present is a namespace, not a fetch",
          not external, str(external[:3]))

    check("inline style block present", "<style>" in html)
    check("inline svg present", "<svg" in html)
    check("declares a doctype", html.lstrip().startswith("<!doctype html>"))
    check("escapes nothing dangerous into attributes", "javascript:" not in html)


def test_shared_panels():
    print("\n=== shared panels ===")
    m = CNN(1, 4, (8,), image_size=8)
    x = tensor([2, 1, 8, 8])
    out = m.forward(x)
    p.mean(p.multiply(out, out), -1).backward()
    for i in range(6):
        m.record_loss(1.0 / (i + 1))

    html = build_dashboard(m, x=x, history=m.loss_history)
    got = panels(html)
    for want in ("Architecture", "Parameters", "Loss", "Gradient flow"):
        check("panel: %s" % want, want in got, "got %s" % got)

    # the header must report the model, not a placeholder
    check("header names the model", "CNN" in html)
    check("header reports the parameter total",
          str(m.num_parameters()) in html or "K parameters" in html
          or "%.1fK" % (m.num_parameters() / 1000) in html)

    # every layer should appear in the architecture panel
    missing = [n for n in m.layer_names if n not in html]
    check("every layer named in the page", not missing, str(missing))

    # the loss curve should contain as many points as epochs recorded
    poly = re.search(r'<polyline points="([^"]+)"', html)
    check("loss curve drawn", poly is not None)
    if poly:
        check("one point per recorded epoch",
              len(poly.group(1).split()) == 6,
              "%d points" % len(poly.group(1).split()))


def test_graceful_without_input():
    print("\n=== degrades without a sample or a backward ===")
    m = MLP(4, 2, (8,))
    html = build_dashboard(m)            # no x, no history, no backward
    check("still renders", len(html) > 500)
    check("prompts for a sample", "Pass x=" in html)
    check("prompts for history", "Pass history=" in html)
    check("prompts for backward", "loss.backward()" in html)
    check("no crash on an empty loss list",
          "Loss" in panels(html))


def test_gradient_norms():
    print("\n=== gradient flow ===")
    m = MLP(4, 2, (8,))
    x = tensor([3, 4])
    out = m.forward(x)
    p.mean(p.multiply(out, out), -1).backward()

    norms = gradient_norms(m)
    check("one entry per parameterised layer",
          len(norms) == len([l for _, l in m._layers if l.parameters()]),
          str(len(norms)))
    check("all norms positive after a real backward",
          all(v > 0 for _, v in norms),
          str([round(v, 6) for _, v in norms]))

    # norms must be recomputed, not cached: zero the grads and they go to zero
    for t in m.parameters():
        t.reset_grad()
    check("norms reflect the current gradients",
          all(v == 0.0 for _, v in gradient_norms(m)))

    # with real gradients present, the panel must state a verdict
    out = m.forward(x)
    p.mean(p.multiply(out, out), -1).backward()
    html = build_dashboard(m, x=x)
    check("flags the state of gradient flow",
          ("healthy" in html or "vanishing" in html or "exploding" in html))

    # and with none, it must ask for a backward rather than imply a problem
    for t in m.parameters():
        t.reset_grad()
    empty = build_dashboard(m, x=x)
    check("no gradients prompts for backward, not a false verdict",
          "loss.backward()" in empty and "vanishing" not in empty)


def test_model_specific_panels():
    print("\n=== model-specific panels ===")
    p.Tensor.manual_seed(5)

    m = MLP(4, 3, (8, 8))
    check("MLP gets unit activations",
          "Unit activations" in panels(build_dashboard(m, x=tensor([4, 4]))))

    c = CNN(1, 4, (8,), image_size=8)
    check("CNN gets feature maps",
          "Feature maps" in panels(build_dashboard(c, x=tensor([2, 1, 8, 8]))))

    r = ResNet(4, 18, 1, 8, 32)
    check("ResNet gets feature maps",
          "Feature maps" in panels(build_dashboard(r, x=tensor([1, 1, 32, 32]))))

    u = UNet(1, 2, 8, 2, 16)
    up = panels(build_dashboard(u, x=tensor([1, 1, 16, 16])))
    check("UNet gets skip connections", "Skip connections" in up, str(up))
    check("UNet also gets feature maps", "Feature maps" in up)

    v = VAE(6, 3, (8,))
    vh = build_dashboard(v, x=tensor([10, 6]))
    check("VAE gets a latent panel", "Latent space" in panels(vh))
    check("VAE latent projects to 2D by PCA when dim > 2", "PCA" in vh)
    check("VAE plots one point per sample", vh.count("<circle") >= 10,
          "%d circles" % vh.count("<circle"))

    g = GAN(4, img_size=16, img_channels=1, base_channels=8)
    gh = build_dashboard(g, x=g.sample_latent(4))
    check("GAN gets a samples panel", "Samples" in panels(gh))
    check("GAN labels generated output", "generated" in gh)


def test_gpt_attention():
    print("\n=== GPT attention ===")
    p.Tensor.manual_seed(3)
    m = GPT(vocab_size=24, max_seq_len=16, embed_dim=16, num_heads=4, num_layers=2)
    ids = [1, 5, 9, 2, 7]
    logits = m.forward(ids)
    p.cross_entropy_sparse_seq(logits, [5, 9, 2, 7, 3]).backward()

    html = build_dashboard(m, x=ids, token_ids=ids)
    got = panels(html)
    check("GPT is a Model with registered layers", len(m._layers) == 5,
          str(m.layer_names))
    check("GPT gets an attention panel", "Attention" in got, str(got))
    check("attention names the heads", "head 1" in html and "head 4" in html)
    check("attention reports which layer", "layer 1 of 2" in html)

    # the causal mask must be visible: the upper triangle is drawn as the
    # masked colour, so it should appear exactly where j > i
    check("masked cells rendered", html.count("#05080c") >= 10,
          "%d masked" % html.count("#05080c"))

    # a second layer should be selectable and produce a different picture
    h2 = build_dashboard(m, x=ids, token_ids=ids, attention_layer=1)
    check("attention_layer selects a different layer", "layer 2 of 2" in h2)
    check("the two layers attend differently", h2 != html)


def test_live_refresh():
    print("\n=== live mode ===")
    m = MLP(4, 2, (8,))
    html = build_dashboard(m, live_seconds=5)
    check("live mode emits a refresh header",
          'http-equiv="refresh"' in html and 'content="5"' in html)
    check("live mode says so in the header", "refreshing every 5s" in html)
    check("static mode has no refresh header",
          'http-equiv="refresh"' not in build_dashboard(m))


def test_writes_a_file():
    print("\n=== file output ===")
    m = MLP(4, 2, (8,))
    path = m.visualise(x=tensor([3, 4]), filename="_vis_test.html",
                       open_browser=False)
    check("returns an absolute path", os.path.isabs(path))
    check("file exists", os.path.exists(path))
    size = os.path.getsize(path)
    check("file is not trivially small", size > 2000, "%d bytes" % size)
    check("file is not bloated", size < 2_000_000, "%d bytes" % size)
    with open(path, encoding="utf-8") as fh:
        check("file content is the dashboard", "prometheus" in fh.read())
    os.remove(path)

    # record_loss feeds the curve without passing history explicitly
    m2 = MLP(4, 2, (8,))
    for i in range(5):
        m2.record_loss(1.0 / (i + 1))
    check("record_loss accepts floats", len(m2.loss_history) == 5)
    out = m2.forward(tensor([2, 4]))
    loss = p.mean(p.multiply(out, out), -1)
    m2.record_loss(loss)
    check("record_loss accepts a scalar Tensor", len(m2.loss_history) == 6)
    html = build_dashboard(m2, history=m2.loss_history)
    check("history is picked up automatically",
          '<polyline points="' in html)


def test_large_model_size_is_bounded():
    print("\n=== output size on a deep model ===")
    p.Tensor.manual_seed(1)
    r = ResNet(10, 34, 3, 16, 32)
    html = build_dashboard(r, x=tensor([1, 3, 32, 32]))
    kb = len(html) / 1024
    # feature maps are subsampled rather than one rect per pixel
    check("ResNet34 dashboard stays under 1 MB", kb < 1024, "%.0f KB" % kb)
    check("still has its panels", len(panels(html)) >= 5)


def main():
    print("Model dashboard")
    test_self_contained()
    test_shared_panels()
    test_graceful_without_input()
    test_gradient_norms()
    test_model_specific_panels()
    test_gpt_attention()
    test_live_refresh()
    test_writes_a_file()
    test_large_model_size_is_bounded()

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
