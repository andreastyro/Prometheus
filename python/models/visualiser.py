"""
Self-contained HTML dashboard for any Prometheus Model.

    model.visualise()                      # architecture + parameters
    model.visualise(x=sample)              # adds activations and the
                                           # model-specific panel
    model.visualise(x=sample, history=losses, live=True)

Emits one .html file with inline SVG and vanilla JS. No server, no CDN, no
external stylesheet: it opens from the filesystem and works offline.

Everything is computed in Python from the public API -- layer names, parameter
tensors, and the activations returned by model.trace(). Nothing here needs a
C++ change, which is the payoff for writing the models in Python.
"""

import math
import os
import tempfile
import webbrowser

import prometheus as p

# ── palette ──────────────────────────────────────────────────────────────────
BG = "#080c12"
PANEL = "#0e141c"
EDGE = "#1b2634"
TEXT = "#d7e0ea"
DIM = "#7d8a9a"
ACCENT = "#ff7a3d"
GOOD = "#49c98c"
WARN = "#e4c25c"
BAD = "#e2605a"

# per layer type, so a diagram is readable at a glance
TYPE_COLOUR = {
    "Linear": "#4f8ff7", "Conv2D": "#ff7a3d", "ConvTranspose2D": "#ff9f6b",
    "MaxPool2D": "#9b6cf2", "AvgPool2D": "#8f7ae0",
    "BatchNorm": "#49c98c", "GroupNorm": "#3fb8a8", "LayerNorm": "#35a3c2",
    "ReLU": "#e4c25c", "Sigmoid": "#d9a84e", "Tanh": "#c9923f",
    "Softmax": "#b87d35", "Dropout": "#6b7786", "Flatten": "#55627a",
    "Identity": "#39424f", "ResidualBlock": "#ff6b9d",
    "Embedding": "#7ad0ff", "TransformerBlock": "#ff5f8d",
    "_ConvBlock": "#ff8f55",
}


def _colour(obj):
    return TYPE_COLOUR.get(type(obj).__name__, "#5a6678")


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _fmt(n):
    if n >= 1_000_000:
        return "%.2fM" % (n / 1_000_000)
    if n >= 1_000:
        return "%.1fK" % (n / 1_000)
    return str(n)


def _flat(t):
    return list(t.data)


def _shape_of(t):
    return list(t.shape)


# ── small SVG helpers ────────────────────────────────────────────────────────
def _svg(w, h, body, extra=""):
    return ('<svg viewBox="0 0 %d %d" width="100%%" height="%d" %s>%s</svg>'
            % (w, h, h, extra, body))


def _text(x, y, s, size=11, fill=TEXT, anchor="start", weight="400"):
    return ('<text x="%.1f" y="%.1f" font-size="%d" fill="%s" '
            'text-anchor="%s" font-weight="%s" '
            'font-family="ui-monospace,SFMono-Regular,Menlo,monospace">%s</text>'
            % (x, y, size, fill, anchor, weight, _esc(s)))


def _rect(x, y, w, h, fill, rx=3, opacity=1.0, stroke=None):
    st = ' stroke="%s"' % stroke if stroke else ""
    return ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%d" '
            'fill="%s" opacity="%.2f"%s/>' % (x, y, max(w, 0.0), max(h, 0.0), rx, fill, opacity, st))


def _heat(v):
    """0..1 -> a blue→orange ramp that reads on a dark background."""
    v = 0.0 if v != v else max(0.0, min(1.0, v))
    if v < 0.5:
        t = v / 0.5
        r, g, b = int(40 + 39 * t), int(80 + 63 * t), int(180 - 20 * t)
    else:
        t = (v - 0.5) / 0.5
        r, g, b = int(79 + 176 * t), int(143 - 21 * t), int(160 - 99 * t)
    return "#%02x%02x%02x" % (r, g, b)


def _subsample(flat, h, w, cap=14):
    """Nearest-neighbour downsample a h*w slice to at most cap*cap.

    One <rect> per pixel is fine for an 8x8 map and ruinous for a 32x32x64
    layer: the ResNet dashboard came out at 2.9 MB before this. Striding keeps
    the picture readable while bounding the file size.
    """
    if h <= cap and w <= cap:
        return flat, h, w
    nh, nw = min(h, cap), min(w, cap)
    out = []
    for r in range(nh):
        sr = r * h // nh
        for c in range(nw):
            sc = c * w // nw
            out.append(flat[sr * w + sc])
    return out, nh, nw


# ── panel 1: architecture ────────────────────────────────────────────────────
def _architecture(model, shapes):
    """Layered graph: one node per layer, shape annotations on the edges."""
    layers = model._layers
    if not layers:
        return "<p>no layers</p>"

    node_h, gap, pad = 34, 16, 22
    label_w = 150
    width = 760
    height = pad * 2 + len(layers) * (node_h + gap)

    out = []
    for i, (name, layer) in enumerate(layers):
        y = pad + i * (node_h + gap)
        n = sum(t.num_el() for t in layer.parameters())
        col = _colour(layer)

        # connecting edge, with the shape produced by the previous layer
        if i > 0:
            out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" '
                       'stroke="%s" stroke-width="1.5"/>'
                       % (pad + 8, y - gap, pad + 8, y, EDGE))
            prev = shapes.get(layers[i - 1][0])
            if prev:
                out.append(_text(pad + 16, y - gap + 11,
                                 "x".join(str(d) for d in prev), 9, DIM))

        out.append(_rect(pad, y, label_w, node_h, col, opacity=0.16))
        out.append(_rect(pad, y, 3, node_h, col))
        out.append(_text(pad + 12, y + 14, name, 11, TEXT, weight="600"))
        out.append(_text(pad + 12, y + 27, type(layer).__name__, 9, DIM))

        if n:
            out.append(_text(pad + label_w + 14, y + 21, _fmt(n) + " params",
                             10, DIM))
        sh = shapes.get(name)
        if sh:
            out.append(_text(width - pad, y + 21,
                             "-> " + "x".join(str(d) for d in sh), 10, col,
                             anchor="end"))
    return _svg(width, height, "".join(out))


# ── panel 2: parameter breakdown ─────────────────────────────────────────────
def _parameters(model):
    rows = [(n, sum(t.num_el() for t in l.parameters())) for n, l in model._layers]
    rows = [(n, v) for n, v in rows if v > 0]
    if not rows:
        return "<p>this model has no parameters</p>"

    peak = max(v for _, v in rows)
    total = sum(v for _, v in rows)
    bar_h, gap, pad, label_w = 18, 7, 16, 150
    width = 760
    height = pad * 2 + len(rows) * (bar_h + gap) + 10

    out = []
    for i, (name, v) in enumerate(rows):
        y = pad + i * (bar_h + gap)
        w = (width - pad * 2 - label_w - 90) * (v / peak)
        out.append(_text(pad, y + 13, name, 10, TEXT))
        out.append(_rect(pad + label_w, y, w, bar_h, ACCENT, opacity=0.85))
        out.append(_text(pad + label_w + w + 8, y + 13,
                         "%s  %.1f%%" % (_fmt(v), 100.0 * v / total), 9, DIM))
    return _svg(width, height, "".join(out))


# ── panel 3: loss curve ──────────────────────────────────────────────────────
def _loss_curve(history):
    if not history or len(history) < 2:
        return ('<p class="hint">Pass history=[...] -- a list of per-epoch '
                'losses -- to draw the curve.</p>')

    w, h, pad = 760, 240, 38
    lo, hi = min(history), max(history)
    span = (hi - lo) or 1.0
    n = len(history)

    def px(i):
        return pad + (w - pad * 2) * (i / max(1, n - 1))

    def py(v):
        return h - pad - (h - pad * 2) * ((v - lo) / span)

    pts = " ".join("%.1f,%.1f" % (px(i), py(v)) for i, v in enumerate(history))
    out = [
        '<defs><linearGradient id="lg" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%%" stop-color="%s" stop-opacity="0.45"/>'
        '<stop offset="100%%" stop-color="%s" stop-opacity="0"/>'
        '</linearGradient></defs>' % (ACCENT, ACCENT),
    ]
    # gridlines and y labels
    for k in range(5):
        v = lo + span * k / 4
        y = py(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
                   'stroke-width="1"/>' % (pad, y, w - pad, y, EDGE))
        out.append(_text(pad - 6, y + 3, "%.4g" % v, 9, DIM, anchor="end"))

    out.append('<polygon points="%s %.1f,%.1f %.1f,%.1f" fill="url(#lg)"/>'
               % (pts, px(n - 1), h - pad, px(0), h - pad))
    out.append('<polyline points="%s" fill="none" stroke="%s" '
               'stroke-width="2.5" stroke-linejoin="round" '
               'stroke-linecap="round"/>' % (pts, ACCENT))
    out.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s"/>'
               % (px(n - 1), py(history[-1]), ACCENT))
    out.append(_text(pad, h - 12, "epoch 1", 9, DIM))
    out.append(_text(w - pad, h - 12, "epoch %d" % n, 9, DIM, anchor="end"))
    out.append(_text(w - pad, pad - 12,
                     "first %.4g   last %.4g" % (history[0], history[-1]),
                     10, TEXT, anchor="end"))
    return _svg(w, h, "".join(out))


# ── panel 4: gradient flow ───────────────────────────────────────────────────
def gradient_norms(model):
    """Per-layer L2 norm of the parameter gradients, after a backward pass."""
    out = []
    for name, layer in model._layers:
        ts = layer.parameters()
        if not ts:
            continue
        total = 0.0
        for t in ts:
            for g in t.grad:
                total += g * g
        out.append((name, math.sqrt(total)))
    return out


def _gradient_flow(norms):
    # All-zero norms means no backward has run yet, which is different from a
    # model with no parameters. Drawing a row of empty bars for that would look
    # like a vanishing-gradient problem rather than missing data.
    if not norms or all(v == 0.0 for _, v in norms):
        return ('<p class="hint">Call loss.backward() before visualise() to '
                'see gradient flow.</p>')

    peak = max(v for _, v in norms) or 1.0
    # flag the classic failure modes
    tiny = [n for n, v in norms if v < peak * 1e-4]
    huge = [n for n, v in norms if v > 1e3]

    bar_h, gap, pad, label_w = 18, 7, 16, 150
    width = 760
    height = pad * 2 + len(norms) * (bar_h + gap) + 34

    out = []
    for i, (name, v) in enumerate(norms):
        y = pad + i * (bar_h + gap)
        frac = v / peak
        w = (width - pad * 2 - label_w - 110) * frac
        col = BAD if v > 1e3 else (WARN if frac < 1e-4 else _heat(frac))
        out.append(_text(pad, y + 13, name, 10, TEXT))
        out.append(_rect(pad + label_w, y, max(w, 1.5), bar_h, col, opacity=0.9))
        out.append(_text(pad + label_w + max(w, 1.5) + 8, y + 13,
                         "%.3e" % v, 9, DIM))

    msg_y = pad + len(norms) * (bar_h + gap) + 14
    if huge:
        out.append(_text(pad, msg_y,
                         "exploding: " + ", ".join(huge[:6]), 10, BAD))
    elif tiny:
        out.append(_text(pad, msg_y,
                         "vanishing (<1e-4 of peak): " + ", ".join(tiny[:6]),
                         10, WARN))
    else:
        out.append(_text(pad, msg_y, "gradient flow looks healthy", 10, GOOD))
    return _svg(width, height, "".join(out))


# ── model-specific panels ────────────────────────────────────────────────────
def _activation_heatmap(trace, limit=6):
    """MLP: one row per neuron-vector activation, colour-coded by value."""
    rows = [(n, t) for n, t in trace if len(_shape_of(t)) == 2][:limit]
    if not rows:
        return '<p class="hint">no 2D activations to show</p>'

    pad, label_w, cell, gap = 16, 90, 15, 10
    width = 760
    height = pad * 2 + len(rows) * (cell + gap) + 16
    out = []
    for i, (name, t) in enumerate(rows):
        y = pad + i * (cell + gap)
        vals = _flat(t)
        sh = _shape_of(t)
        # average over the batch so the row is per neuron
        batch, feat = sh[0], sh[1]
        per = [sum(vals[b * feat + f] for b in range(batch)) / batch
               for f in range(feat)]
        lo, hi = min(per), max(per)
        span = (hi - lo) or 1.0
        cw = min(cell, (width - pad * 2 - label_w) / max(1, feat))
        out.append(_text(pad, y + 11, name, 10, TEXT))
        for f, v in enumerate(per):
            out.append(_rect(pad + label_w + f * cw, y, cw - 1, cell,
                             _heat((v - lo) / span), rx=2))
        out.append(_text(pad + label_w + feat * cw + 8, y + 11,
                         "%d units" % feat, 9, DIM))
    out.append(_text(pad, height - 8,
                     "each cell is one unit, averaged over the batch; "
                     "blue low, orange high", 9, DIM))
    return _svg(width, height, "".join(out))


def _feature_maps(trace, max_layers=4, max_channels=6):
    """CNN/ResNet/UNet: a tiled grid of channel activations per conv layer."""
    convs = [(n, t) for n, t in trace if len(_shape_of(t)) == 4][:max_layers]
    if not convs:
        return '<p class="hint">no 4D feature maps to show</p>'

    pad, label_h, tile_gap = 16, 16, 6
    width = 760
    tile = 52
    height = pad * 2 + len(convs) * (tile + label_h + tile_gap + 10)

    out = []
    y = pad
    for name, t in convs:
        n, c, hh, ww = _shape_of(t)
        vals = _flat(t)
        out.append(_text(pad, y + 11, "%s   %dx%dx%d" % (name, c, hh, ww),
                         10, TEXT))
        y += label_h
        shown = min(c, max_channels)
        for ch in range(shown):
            base = ch * hh * ww          # batch item 0
            sl, sh_, sw_ = _subsample(vals[base:base + hh * ww], hh, ww)
            lo, hi = min(sl), max(sl)
            span = (hi - lo) or 1.0
            px = tile / max(sh_, sw_)
            ox = pad + ch * (tile + tile_gap)
            out.append(_rect(ox, y, tile, tile, "#05080c", rx=2))
            for r in range(sh_):
                for cc in range(sw_):
                    v = (sl[r * sw_ + cc] - lo) / span
                    out.append(_rect(ox + cc * px, y + r * px, px + 0.5,
                                     px + 0.5, _heat(v), rx=0))
        if c > shown:
            out.append(_text(pad + shown * (tile + tile_gap) + 6, y + tile / 2,
                             "+%d more" % (c - shown), 9, DIM))
        y += tile + tile_gap + 10
    return _svg(width, height, "".join(out))


def _unet_skips(model, shapes):
    """UNet: which encoder level feeds which decoder level."""
    pairs = model.skip_pairs()
    if not pairs:
        return '<p class="hint">no skip connections</p>'

    pad, box_w, box_h, gap = 24, 150, 34, 22
    width, left, right = 760, 24, 760 - 24 - 150
    height = pad * 2 + len(pairs) * (box_h + gap) + box_h + gap

    out = []
    for i, (enc, dec) in enumerate(pairs):
        y = pad + i * (box_h + gap)
        out.append(_rect(left, y, box_w, box_h, TYPE_COLOUR["Conv2D"], opacity=0.18))
        out.append(_rect(left, y, 3, box_h, TYPE_COLOUR["Conv2D"]))
        out.append(_text(left + 12, y + 15, enc, 11, TEXT, weight="600"))
        sh = shapes.get(enc)
        if sh:
            out.append(_text(left + 12, y + 28,
                             "x".join(str(d) for d in sh), 9, DIM))

        out.append(_rect(right, y, box_w, box_h, TYPE_COLOUR["ConvTranspose2D"],
                         opacity=0.18))
        out.append(_rect(right, y, 3, box_h, TYPE_COLOUR["ConvTranspose2D"]))
        out.append(_text(right + 12, y + 15, dec, 11, TEXT, weight="600"))
        sh = shapes.get(dec)
        if sh:
            out.append(_text(right + 12, y + 28,
                             "x".join(str(d) for d in sh), 9, DIM))

        # the skip itself
        mid = y + box_h / 2
        out.append('<path d="M %d %.1f C %d %.1f, %d %.1f, %d %.1f" '
                   'fill="none" stroke="%s" stroke-width="2" '
                   'stroke-dasharray="5 4"/>'
                   % (left + box_w, mid, left + box_w + 120, mid - 26,
                      right - 120, mid - 26, right, mid, ACCENT))
        out.append(_text((left + box_w + right) / 2, mid - 30,
                         "concat on channels", 9, ACCENT, anchor="middle"))

    y = pad + len(pairs) * (box_h + gap)
    out.append(_rect(left + (right - left) / 2 - 20, y, box_w, box_h,
                     "#9b6cf2", opacity=0.18))
    out.append(_text(left + (right - left) / 2 - 8, y + 21, "bottleneck", 11,
                     TEXT, weight="600"))
    return _svg(width, height, "".join(out))


def _pca_2d(rows):
    """Project to 2D by power iteration. Avoids a numpy dependency."""
    n, d = len(rows), len(rows[0])
    mean = [sum(r[j] for r in rows) / n for j in range(d)]
    cen = [[r[j] - mean[j] for j in range(d)] for r in rows]

    def top_component(data, exclude=None):
        v = [1.0 / math.sqrt(d)] * d
        for _ in range(60):
            acc = [0.0] * d
            for r in data:
                dot = sum(r[j] * v[j] for j in range(d))
                for j in range(d):
                    acc[j] += dot * r[j]
            if exclude:
                dot = sum(acc[j] * exclude[j] for j in range(d))
                acc = [acc[j] - dot * exclude[j] for j in range(d)]
            norm = math.sqrt(sum(a * a for a in acc)) or 1.0
            v = [a / norm for a in acc]
        return v

    c1 = top_component(cen)
    c2 = top_component(cen, exclude=c1)
    return [(sum(r[j] * c1[j] for j in range(d)),
             sum(r[j] * c2[j] for j in range(d))) for r in cen]


def _vae_latent(model, x, labels=None):
    """VAE: the encoded means, projected to 2D if the latent is bigger."""
    mu, _ = model.encode(x)
    sh = _shape_of(mu)
    vals = _flat(mu)
    batch, dim = sh[0], sh[1]
    rows = [[vals[b * dim + j] for j in range(dim)] for b in range(batch)]
    if batch < 2:
        return ('<p class="hint">Pass a batch of at least 2 samples to plot '
                'the latent space.</p>')

    pts = rows if dim == 2 else _pca_2d(rows)
    w, h, pad = 760, 320, 40
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xs_span = (xhi - xlo) or 1.0
    ys_span = (yhi - ylo) or 1.0

    palette = ["#ff7a3d", "#4f8ff7", "#49c98c", "#e4c25c", "#9b6cf2",
               "#ff6b9d", "#3fb8a8", "#d9a84e"]
    out = ['<rect x="%d" y="%d" width="%d" height="%d" fill="none" '
           'stroke="%s"/>' % (pad, pad / 2, w - pad * 2, h - pad, EDGE)]
    for i, (qx, qy) in enumerate(pts):
        cx = pad + (w - pad * 2) * ((qx - xlo) / xs_span)
        cy = (h - pad / 2) - (h - pad) * ((qy - ylo) / ys_span)
        col = palette[(labels[i] % len(palette))] if labels else ACCENT
        out.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s" '
                   'opacity="0.85"/>' % (cx, cy, col))
    out.append(_text(pad, h - 8,
                     "latent dim %d%s, %d samples"
                     % (dim, "" if dim == 2 else " projected to 2D by PCA", batch),
                     9, DIM))
    return _svg(w, h, "".join(out))


def _image_grid(tensor, title, max_n=8):
    """Render [batch, channels, h, w] as a row of tiles."""
    sh = _shape_of(tensor)
    if len(sh) != 4:
        return ""
    n, c, hh, ww = sh
    vals = _flat(tensor)
    shown = min(n, max_n)
    tile, gap, pad = 70, 8, 16
    width = 760
    height = pad * 2 + tile + 26
    out = [_text(pad, pad + 10, title, 10, TEXT)]
    for i in range(shown):
        ox = pad + i * (tile + gap)
        oy = pad + 18
        out.append(_rect(ox, oy, tile, tile, "#05080c", rx=2))
        # average the channels so a colour image renders as luminance
        lum = []
        for r in range(hh):
            for cc in range(ww):
                acc = 0.0
                for ch in range(c):
                    acc += vals[((i * c + ch) * hh + r) * ww + cc]
                lum.append(acc / c)
        sl, sh_, sw_ = _subsample(lum, hh, ww, cap=20)
        px = tile / max(sh_, sw_)
        for r in range(sh_):
            for cc in range(sw_):
                v = (sl[r * sw_ + cc] + 1.0) / 2.0   # tanh output is [-1, 1]
                out.append(_rect(ox + cc * px, oy + r * px, px + 0.5,
                                 px + 0.5, _heat(v), rx=0))
    return _svg(width, height, "".join(out))


def _gan_samples(model, real=None):
    z = model.sample_latent(8)
    fake = model.forward(z)
    html = _image_grid(fake, "generated")
    if real is not None:
        html += _image_grid(real, "real")
    else:
        html += ('<p class="hint">Pass real=&lt;batch tensor&gt; to compare '
                 'against real samples.</p>')
    return html


def _gpt_attention(model, token_ids, layer=0, max_heads=4):
    """Recompute attention weights from the exposed projection matrices.

    The C++ MultiHeadAttention does not return its weights, but W_q and W_k are
    bound, so the pattern can be reproduced in Python without touching C++:
    softmax(QK^T / sqrt(head_dim)) with the causal mask applied.
    """
    blocks = getattr(model, "blocks", None)
    if not blocks:
        return '<p class="hint">this model has no transformer blocks</p>'
    layer = max(0, min(layer, len(blocks) - 1))
    attn = blocks[layer].attn

    seq = len(token_ids)
    emb = model.embed_dim
    heads = model.num_heads
    hd = emb // heads

    ids = p.Tensor([1, seq], [float(t) for t in token_ids])
    x = p.reshape_op(model.tok_emb.forward(ids), [seq, emb])
    if not getattr(model, "rope", False):
        pos = p.Tensor([1, seq], [float(i) for i in range(seq)])
        x = p.add(x, p.reshape_op(model.pos_emb.forward(pos), [seq, emb]))

    q = _flat(p.matmul(x, attn.W_q))
    k = _flat(p.matmul(x, attn.W_k))

    pad, cell, gap = 16, 26, 22
    shown = min(heads, max_heads)
    width = 760
    grid = cell * seq
    height = pad * 2 + 20 + grid + gap

    out = [_text(pad, pad + 10, "layer %d of %d, softmax(QK^T/sqrt(d)) with the "
                 "causal mask" % (layer + 1, len(blocks)), 10, TEXT)]
    for hh in range(shown):
        ox = pad + hh * (grid + gap)
        if ox + grid > width - pad:
            break
        for i in range(seq):
            scores = []
            for j in range(seq):
                if j > i:
                    scores.append(None)       # masked
                    continue
                dot = 0.0
                for d in range(hd):
                    dot += q[i * emb + hh * hd + d] * k[j * emb + hh * hd + d]
                scores.append(dot / math.sqrt(hd))
            real = [s for s in scores if s is not None]
            m = max(real)
            exps = [math.exp(s - m) if s is not None else 0.0 for s in scores]
            tot = sum(exps) or 1.0
            for j in range(seq):
                out.append(_rect(ox + j * cell, pad + 20 + i * cell,
                                 cell - 1, cell - 1,
                                 "#05080c" if scores[j] is None
                                 else _heat(exps[j] / tot), rx=2))
        out.append(_text(ox, pad + 20 + grid + 12, "head %d" % (hh + 1), 9, DIM))
    return _svg(width, height, "".join(out))


# ── assembly ─────────────────────────────────────────────────────────────────
def _panel(title, body, note=""):
    n = '<div class="note">%s</div>' % _esc(note) if note else ""
    return ('<section><h2>%s</h2>%s<div class="body">%s</div></section>'
            % (_esc(title), n, body))


_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:%(bg)s;color:%(text)s;font-family:ui-monospace,SFMono-Regular,
Menlo,monospace;padding:28px;line-height:1.5}
header{margin-bottom:24px;border-bottom:1px solid %(edge)s;padding-bottom:18px}
h1{font-size:22px;font-weight:600;letter-spacing:-0.3px}
h1 span{color:%(accent)s}
.meta{color:%(dim)s;font-size:12px;margin-top:8px}
.meta b{color:%(text)s;font-weight:600}
section{background:%(panel)s;border:1px solid %(edge)s;border-radius:8px;
padding:18px;margin-bottom:18px}
h2{font-size:13px;font-weight:600;color:%(text)s;margin-bottom:4px;
text-transform:uppercase;letter-spacing:0.8px}
.note{font-size:11px;color:%(dim)s;margin-bottom:12px}
.hint{font-size:12px;color:%(dim)s;padding:14px 0}
.body{overflow-x:auto}
footer{color:%(dim)s;font-size:11px;margin-top:8px;text-align:center}
""" % {"bg": BG, "panel": PANEL, "edge": EDGE, "text": TEXT, "dim": DIM,
       "accent": ACCENT}


def build_dashboard(model, x=None, history=None, real=None, labels=None,
                    token_ids=None, attention_layer=0, live_seconds=None):
    """Return the dashboard as one self-contained HTML string."""
    kind = type(model).__name__

    shapes, trace = {}, []
    if x is not None:
        trace = model.trace(x)
        for name, t in trace:
            shapes[name] = _shape_of(t)

    parts = [_panel("Architecture", _architecture(model, shapes),
                    "one node per layer, output shape on the right")]
    parts.append(_panel("Parameters", _parameters(model),
                        "%s total across %d layers"
                        % (_fmt(model.num_parameters()), len(model._layers))))
    parts.append(_panel("Loss", _loss_curve(history),
                        "per-epoch training loss"))
    parts.append(_panel("Gradient flow", _gradient_flow(gradient_norms(model)),
                        "L2 norm of each layer's parameter gradients after the "
                        "last backward pass"))

    # ---- the model-specific panel ----
    if trace:
        if kind == "MLP":
            parts.append(_panel("Unit activations", _activation_heatmap(trace),
                                "which units respond to this input"))
        elif kind in ("CNN", "ResNet"):
            parts.append(_panel("Feature maps", _feature_maps(trace),
                                "channel activations from one forward pass"))
        elif kind == "UNet":
            parts.append(_panel("Skip connections", _unet_skips(model, shapes),
                                "which encoder level feeds which decoder level"))
            parts.append(_panel("Feature maps", _feature_maps(trace)))
        elif kind == "VAE":
            parts.append(_panel("Latent space", _vae_latent(model, x, labels),
                                "encoded means for this batch"))
        elif kind in ("GAN", "Generator"):
            parts.append(_panel("Samples", _gan_samples(model, real),
                                "generated against real"))
        elif kind == "GPT" and token_ids:
            parts.append(_panel("Attention", _gpt_attention(model, token_ids,
                                                            attention_layer),
                                "token-to-token weights per head"))
    else:
        parts.append(_panel("Activations",
                            '<p class="hint">Pass x=&lt;a sample batch&gt; to '
                            'visualise() for activations, feature maps and '
                            'shape annotations.</p>'))

    refresh = ('<meta http-equiv="refresh" content="%d">' % live_seconds
               if live_seconds else "")
    return ("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>%(kind)s - prometheus</title>%(refresh)s
<style>%(css)s</style></head><body>
<header>
  <h1>%(kind)s <span>/ prometheus</span></h1>
  <div class="meta"><b>%(params)s</b> parameters &middot; <b>%(layers)d</b>
  layers &middot; mode <b>%(mode)s</b>%(livenote)s</div>
</header>
%(parts)s
<footer>generated by prometheus &middot; self-contained, no external
dependencies</footer>
</body></html>""" % {
        "kind": _esc(kind),
        "refresh": refresh,
        "css": _CSS,
        "params": _fmt(model.num_parameters()),
        "layers": len(model._layers),
        "mode": "training" if model.training else "eval",
        "livenote": (" &middot; live, refreshing every %ds" % live_seconds
                     if live_seconds else ""),
        "parts": "".join(parts),
    })


def visualise(model, x=None, history=None, filename=None, open_browser=True,
              real=None, labels=None, token_ids=None, attention_layer=0,
              live_seconds=None):
    """Write the dashboard to disk and open it. Returns the path."""
    html = build_dashboard(model, x=x, history=history, real=real,
                           labels=labels, token_ids=token_ids,
                           attention_layer=attention_layer,
                           live_seconds=live_seconds)
    if filename is None:
        fd, filename = tempfile.mkstemp(
            prefix="prometheus_%s_" % type(model).__name__.lower(),
            suffix=".html")
        os.close(fd)
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(html)
    if open_browser:
        webbrowser.open("file://" + os.path.abspath(filename))
    return os.path.abspath(filename)
