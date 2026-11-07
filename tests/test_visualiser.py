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
from python.models.visualiser import (                          # noqa: E402
    activation_stats, build_dashboard, diagnose, gradient_norms,
    run_gradcheck, update_ratios)

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


def test_architecture_toggle():
    print("\n=== 3D / 2D toggle ===")
    m = CNN(1, 4, (8,), image_size=8)
    html = build_dashboard(m, x=tensor([2, 1, 8, 8]))

    check("two toggle buttons", html.count("<button") == 2)
    check("3D view container present", 'id="arch3d"' in html)
    check("2D view container present", 'id="arch2d"' in html)
    check("2D starts hidden", 'id="arch2d" hidden' in html)
    check("3D starts active", 'id="b3d" class="on"' in html)
    check("switcher is defined", "function archView" in html)

    # both views must actually be in the page: it is a static file, so there is
    # nothing to fetch the other one from later
    iso = html[html.index('id="arch3d"'):html.index('id="arch2d"')]
    flat = html[html.index('id="arch2d"'):]
    check("3D view holds the isometric blocks", iso.count("<polygon") > 5,
          "%d polygons" % iso.count("<polygon"))
    check("2D view holds the layered graph", "<svg" in flat[:4000])

    # every layer should be named in both
    missing_3d = [n for n in m.layer_names if n not in iso]
    check("3D names every layer", not missing_3d, str(missing_3d))

    # a model the isometric renderer cannot handle still gets one usable view
    html2 = build_dashboard(MLP(4, 2, (8,)))
    check("still renders without a sample", "Architecture" in html2)


def test_diagnostics():
    print("\n=== diagnostics ===")
    p.Tensor.manual_seed(4)
    m = CNN(1, 4, (8,), image_size=8)
    # batch of 8: the dead-unit measure needs enough samples to be meaningful
    # and reports None below MIN_DEAD_BATCH, since with a batch of 2 a healthy
    # ReLU is zero in both samples a quarter of the time by luck
    x = tensor([8, 1, 8, 8])
    out = m.forward(x)
    p.mean(p.multiply(out, out), -1).backward()
    tr = m.trace(x)

    stats = activation_stats(tr, m)
    check("one stat row per traced layer", len(stats) == len(tr))
    check("zero fraction is a proportion",
          all(0.0 <= st["zeros"] <= 1.0 for st in stats))

    # A ReLU zeroing about half its values is doing its job. The flaggable
    # condition is a unit zero for every sample in the batch, which gets no
    # gradient from any of them. Warning on the former flagged every healthy
    # ReLU in the library.
    relu_stats = [st for st in stats if st["kind"] == "ReLU"]
    check("ReLU zeroes roughly half its values",
          relu_stats and all(0.2 < st["zeros"] < 0.8 for st in relu_stats),
          str([round(st["zeros"], 2) for st in relu_stats]))
    check("dead-unit measure is available at this batch size",
          relu_stats and all(st["dead_units"] is not None for st in relu_stats))
    check("but almost no units are dead across the batch",
          relu_stats and all(st["dead_units"] < 0.25 for st in relu_stats),
          str([round(st["dead_units"], 2) for st in relu_stats]))

    # and it must refuse to answer on a batch too small to mean anything
    small = activation_stats(m.trace(tensor([2, 1, 8, 8])), m)
    check("dead-unit measure withheld below the minimum batch",
          all(st["dead_units"] is None for st in small))
    check("a healthy ReLU is therefore not flagged",
          not [i for i in diagnose(m, tr, None) if "units are zero" in i[2]])

    # saturation only applies where the activation is bounded -- reporting a
    # conv output above 0.99 as saturated flagged every healthy conv layer in
    # the first version of this panel
    unbounded = [st for st in stats if st["kind"] not in ("Sigmoid", "Tanh",
                                                          "Softmax")]
    check("saturation is not reported for unbounded layers",
          all(st["saturated"] is None for st in unbounded),
          str([st["name"] for st in unbounded if st["saturated"] is not None]))

    # ReLU should have some exact zeros, and that is where "dead" is meaningful
    check("dead-unit measure is per unit, not per value",
          all(st["dead_units"] is None or st["dead_units"] <= st["zeros"] + 1e-9
              for st in stats))

    # update ratio must scale with the learning rate. Layers whose only
    # parameters are a bias or a normalisation scale report None: the heuristic
    # divides by the weight norm, and those start at zero.
    r_small = dict((n, v) for n, v, _, _ in update_ratios(m, 1e-5))
    r_big = dict((n, v) for n, v, _, _ in update_ratios(m, 1e-1))
    comparable = [k for k in r_small
                  if r_small[k] is not None and r_small[k] > 0
                  and r_big.get(k) is not None]
    check("some layers have a comparable ratio", comparable, "none")
    check("update ratio scales with lr",
          all(r_big[k] > r_small[k] * 100 for k in comparable))
    check("layers with only bias or scale report no ratio",
          any(v is None for v in r_small.values()),
          "every layer reported a ratio")

    # and a large lr must produce warnings a small one does not
    issues_big = diagnose(m, tr, 1e-1)
    issues_small = diagnose(m, tr, 1e-6)
    big_ratio = [i for i in issues_big if "ratio" in i[2]]
    check("a large lr is flagged", big_ratio, "none flagged")
    check("a tiny lr is not flagged for being too large",
          not [i for i in issues_small if "step is large" in i[2]])

    # non-finite values must be reported as errors, not warnings
    bad = m.parameters()[0]
    saved = list(bad.data)
    d = list(bad.data)
    d[0] = float("nan")
    bad.data = d
    errs = [i for i in diagnose(m, None, None) if i[0] == "error"]
    check("a NaN weight is an error", errs, "not reported")
    bad.data = saved

    html = build_dashboard(m, x=x, lr=1e-3)
    check("diagnostics panel present", "Diagnostics" in panels(html))
    check("panel states a verdict",
          "no problems detected" in html or "warning" in html
          or "error" in html)


def test_gradient_connectivity():
    print("\n=== gradient connectivity ===")
    p.Tensor.manual_seed(4)

    # a correctly trained model must come back clean: a panel that cries wolf
    # on a healthy model is worse than no panel
    m = CNN(1, 4, (8,), image_size=8)
    x = tensor([2, 1, 8, 8])
    opt = p.Adam(m.parameters(), 0.01)
    for _ in range(4):
        opt.zero_grad()
        o = m.forward(x)
        p.mean(p.multiply(o, o), -1).backward()
        opt.step()

    res = run_gradcheck(m, x)
    check("every parameter reported", len(res) > 0)
    check("no false DETACHED on a trained model",
          not [r for r in res if r[1] is False],
          str([r for r in res if r[1] is False][:2]))
    check("no false DEAD on a trained model",
          not [r for r in res if r[1] is None and "no effect" in (r[2] or "")])

    # a parameter genuinely left out of forward() must be caught. gradcheck
    # alone cannot: both its analytical and numeric gradients are zero, so they
    # agree. The influence probe is what notices.
    class Skipped(MLP):
        def forward(self, xx):
            for _, layer in self._layers[2:]:
                xx = layer.forward(xx)
            return xx

    b = Skipped(4, 2, (4,))
    rb = run_gradcheck(b, tensor([3, 4]))
    dead = [r for r in rb if r[1] is None and "no effect" in (r[2] or "")]
    check("a skipped layer is reported as dead weight", len(dead) == 2,
          "%d flagged" % len(dead))

    html = build_dashboard(b, x=tensor([3, 4]))
    check("panel names the condition", "no effect on the output" in html
          or "DEAD WEIGHT" in html)
    check("panel says it is a connectivity check, not precision",
          "connectivity check, not a precision one" in html)


def main():
    print("Model dashboard")
    test_self_contained()
    test_shared_panels()
    test_architecture_toggle()
    test_graceful_without_input()
    test_gradient_norms()
    test_model_specific_panels()
    test_diagnostics()
    test_gradient_connectivity()
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
