"""
prometheus 3D visualiser — PlotNeuralNet-style isometric block diagram.
Layers are arranged left-to-right as 3D cuboids. Height and depth of each
block scale with the layer's output size; colours are assigned by layer type.

Usage:
    from python.visualise import visualise
    visualise(model)                        # opens in browser
    visualise(model, filename="arch.html")  # save to path
"""

import os, math, webbrowser, tempfile

# ── Isometric projection constants ────────────────────────────────────────────
_ANG = 30
_COS = math.cos(math.radians(_ANG))   # ≈ 0.866  (depth → right)
_SIN = math.sin(math.radians(_ANG))   # ≈ 0.500  (depth → up)

# ── Layout ────────────────────────────────────────────────────────────────────
_GAP     = 14    # horizontal gap between blocks
_FW      = 16    # front face width — normal layers
_FW_THIN = 5     # front face width — thin layers (activations, norms)
_PAD_L   = 50
_PAD_R   = 56
_PAD_T   = 72
_PAD_B   = 24
_LABEL_H = 58

# Thin layers: same h/d as neighbour, just very narrow front face
_THIN = {
    'ReLU', 'GELU', 'Sigmoid', 'Tanh', 'Softmax',
    'LayerNorm', 'BatchNorm', 'GroupNorm',
    'Dropout', 'Flatten', 'PositionalEncoding',
}

# ── Colour palette ────────────────────────────────────────────────────────────
_C = {
    'Linear':             '#3D7FBF',
    'Conv2D':             '#D4691A',
    'ConvTranspose2D':    '#E8921E',
    'MaxPool2D':          '#A93226',
    'AvgPool2D':          '#C0392B',
    'Flatten':            '#7F8C8D',
    'Dropout':            '#566573',
    'ReLU':               '#1E8449',
    'GELU':               '#239B56',
    'Sigmoid':            '#148F77',
    'Tanh':               '#117A65',
    'Softmax':            '#1E8449',
    'RNN':                '#6C3483',
    'LSTM':               '#7D3C98',
    'GRU':                '#8E44AD',
    'Embedding':          '#154360',
    'LayerNorm':          '#9A7D0A',
    'BatchNorm':          '#B7950B',
    'GroupNorm':          '#CA6F1E',
    'MultiHeadAttention': '#922B21',
    'TransformerBlock':   '#7B241C',
    'PositionalEncoding': '#2E4057',
    'ResidualBlock':      '#B03A2E',
    'Sequential':         '#5D8AA8',
    'GPT':                '#6C3483',
    'default':            '#4A5568',
}

def _col(tn, palette=None):
    p = palette if palette is not None else _C
    return p.get(tn, p.get('default', _C['default']))

# ── Themes ────────────────────────────────────────────────────────────────────
_THEMES = {
    'default': {
        'bg':         '#080c12',
        'dot_col':    '#ffffff',
        'dot_op':     '0.04',
        'conn':       '#243a52',
        'title':      '#cbd5e1',
        'subtitle':   '#1e3a5a',
        'lbl_attr':   '#334155',
        'lbl_params': '#1a2f45',
        'layers':     _C,
    },
    'mbdtf': {
        'bg':         '#080205',
        'dot_col':    '#8B0018',
        'dot_op':     '0.09',
        'conn':       '#3D0A10',
        'title':      '#E01428',
        'subtitle':   '#5C0A14',
        'lbl_attr':   '#7B1020',
        'lbl_params': '#4A0810',
        'layers': {
            'Linear':             '#C01428',
            'Conv2D':             '#A81220',
            'ConvTranspose2D':    '#B41224',
            'MaxPool2D':          '#801018',
            'AvgPool2D':          '#901018',
            'Flatten':            '#5C0A12',
            'Dropout':            '#3D0608',
            'ReLU':               '#D41C30',
            'GELU':               '#CC1828',
            'Sigmoid':            '#B81420',
            'Tanh':               '#AC1018',
            'Softmax':            '#D41C30',
            'RNN':                '#6B0C14',
            'LSTM':               '#780C18',
            'GRU':                '#880E1C',
            'Embedding':          '#4A0810',
            'LayerNorm':          '#DC1C34',
            'BatchNorm':          '#D01828',
            'GroupNorm':          '#C41424',
            'MultiHeadAttention': '#980E1C',
            'TransformerBlock':   '#880A18',
            'PositionalEncoding': '#600A12',
            'ResidualBlock':      '#B01020',
            'Sequential':         '#580810',
            'GPT':                '#6C0A14',
            'default':            '#4A0810',
        },
    },
}

# ── Colour math ───────────────────────────────────────────────────────────────
def _h2r(h):
    h = h.lstrip('#')
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def _r2h(r,g,b):
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'

def _lighten(h, f=0.30):
    r,g,b = _h2r(h)
    return _r2h(r+(255-r)*f, g+(255-g)*f, b+(255-b)*f)

def _darken(h, f=0.32):
    r,g,b = _h2r(h)
    return _r2h(r*(1-f), g*(1-f), b*(1-f))

# ── Module detection ──────────────────────────────────────────────────────────
def _is_mod(o):
    return (not isinstance(o,type) and
            hasattr(o,'forward') and hasattr(o,'parameters') and
            callable(o.forward) and callable(o.parameters))

def _n_params(m):
    try: return sum(t.num_el() for t in m.parameters())
    except: return 0

def _fmt(n):
    if n >= 1_000_000: return f'{n/1_000_000:.1f}M'
    if n >= 1_000:     return f'{n/1_000:.1f}K'
    return str(n) if n > 0 else '—'

def _collect(model):
    seen, out = set(), []
    def _w(obj, pre=''):
        if id(obj) in seen: return
        seen.add(id(obj))
        try: attrs = vars(obj)
        except TypeError: return
        for k, v in attrs.items():
            if k.startswith('_'): continue
            if _is_mod(v):
                name = f'{pre}{k}' if pre else k
                out.append((name, v));  _w(v, name+'.')
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if _is_mod(item):
                        name = f'{pre}{k}[{i}]'
                        out.append((name, item));  _w(item, name+'.')
    _w(model)
    return out

# ── Block sizing ──────────────────────────────────────────────────────────────
def _lg(v, lo, hi, ref=6):
    """Log-scale v (param count) → [lo, hi]. ref = log10 of expected max."""
    return int(lo + min(1.0, math.log10(max(1,v)) / ref) * (hi - lo))

def _raw_dims(module):
    """(height_px, depth_px, front_width_px). height=0 means inherit."""
    n  = type(module).__name__
    ps = _n_params(module)

    if n in _THIN:
        return 0, 0, _FW_THIN   # inherit h/d from neighbour

    try:
        if n == 'Linear':
            out = module.weights.shape[1]
            h = max(36, min(200, out // 2 + 36))
            d = max(16, min(70,  out // 8 + 16))
            return h, d, _FW

        if n in ('Conv2D', 'ConvTranspose2D'):
            oc = module.out_channels
            h  = max(56, min(230, oc * 2 + 56))
            d  = max(18, min(80,  oc // 2 + 18))
            return h, d, _FW

        if n in ('MaxPool2D', 'AvgPool2D'):
            return 76, 28, _FW

        if n in ('LSTM', 'GRU', 'RNN'):
            h = _lg(ps, 58, 180)
            return h, max(20, h // 3), _FW

        if n == 'Embedding':
            edim = module.embed_dim
            h  = max(68, min(205, edim * 2 + 36))
            dp = max(18, min(70,  edim // 3 + 14))
            return h, dp, _FW

        if n == 'MultiHeadAttention':
            h = _lg(ps, 78, 188)
            return h, max(24, h // 2), _FW

        if n in ('TransformerBlock', 'ResidualBlock'):
            h = _lg(ps, 86, 198)
            return h, max(28, h // 2), _FW

        if n == 'GPT':
            h = _lg(ps, 115, 225)
            return h, max(40, h * 2 // 3), _FW

    except Exception:
        pass

    h = _lg(max(1, ps), 48, 180)
    return h, max(16, h // 3), _FW

def _compute_dims(layers):
    raw = [_raw_dims(m) for _, m in layers]
    last_h, last_d = 120, 70
    dims = []
    for h, d, fw in raw:
        if h == 0:                           # thin — inherit
            dims.append((last_h, last_d, fw))
        else:
            last_h, last_d = h, d
            dims.append((h, d, fw))
    return dims

# ── 3D block drawing ──────────────────────────────────────────────────────────
def _pts(ps):
    return ' '.join(f'{x:.1f},{y:.1f}' for x, y in ps)

def _block(bx, by, h, d, fw, color):
    """Draw one 3D cuboid. by = screen y of front-face bottom."""
    dx, dy   = d * _COS, d * _SIN
    c_front  = color
    c_top    = _lighten(color, 0.28)
    c_side   = _darken(color,  0.30)
    stroke   = _darken(color,  0.48)
    sw       = '0.7'

    # Front face corners
    bl = (bx,      by)
    br = (bx + fw, by)
    tr = (bx + fw, by - h)
    tl = (bx,      by - h)

    # Back corners (shifted +dx, −dy in SVG space)
    br_b = (bx + fw + dx, by      - dy)
    tr_b = (bx + fw + dx, by - h  - dy)
    tl_b = (bx + dx,      by - h  - dy)

    return '\n'.join([
        # Right side face (depth face, darker)
        f'<polygon points="{_pts([br,tr,tr_b,br_b])}" '
        f'fill="{c_side}" stroke="{stroke}" stroke-width="{sw}"/>',
        # Top face (lighter)
        f'<polygon points="{_pts([tl,tr,tr_b,tl_b])}" '
        f'fill="{c_top}" stroke="{stroke}" stroke-width="{sw}"/>',
        # Front face (base color, drawn last → on top)
        f'<polygon points="{_pts([bl,br,tr,tl])}" '
        f'fill="{c_front}" stroke="{stroke}" stroke-width="{sw}"/>',
    ])

# ── Full SVG ──────────────────────────────────────────────────────────────────
def _make_svg(model_name, layers, theme='default'):
    t    = _THEMES.get(theme, _THEMES['default'])
    pal  = t['layers']
    dims = _compute_dims(layers)

    # Centered layout: all blocks share the same vertical centre
    max_above = max(h / 2 + d * _SIN for h, d, _ in dims)
    max_below = max(dim[0] / 2         for dim in dims)
    y_center  = _PAD_T + max_above + 12

    xs, cx = [], _PAD_L
    for h, d, fw in dims:
        xs.append(cx)
        cx += fw + d * _COS + _GAP

    W = cx - _GAP + _PAD_R
    H = y_center + max_below + _LABEL_H + _PAD_B

    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W:.0f}" height="{H:.0f}" '
        f'viewBox="0 0 {W:.0f} {H:.0f}">'
    )

    # ── Defs ─────────────────────────────────────────────────────────────────
    out.append(f'''  <defs>
    <pattern id="dots" width="40" height="40" patternUnits="userSpaceOnUse">
      <circle cx="20" cy="20" r="0.7" fill="{t['dot_col']}" opacity="{t['dot_op']}"/>
    </pattern>
  </defs>''')

    # ── Background ────────────────────────────────────────────────────────────
    out.append(f'  <rect width="{W:.0f}" height="{H:.0f}" fill="{t["bg"]}"/>')
    out.append(f'  <rect width="{W:.0f}" height="{H:.0f}" fill="url(#dots)"/>')

    # ── Title ─────────────────────────────────────────────────────────────────
    tx      = W / 2
    total_p = sum(_n_params(m) for _, m in layers)
    out.append(f'''  <text x="{tx:.0f}" y="32" text-anchor="middle"
      font-family="ui-monospace,SFMono-Regular,Menlo,monospace"
      font-size="14" font-weight="700" fill="{t['title']}" letter-spacing="3">
    {model_name.upper()}
  </text>
  <text x="{tx:.0f}" y="50" text-anchor="middle"
      font-family="ui-monospace,SFMono-Regular,Menlo,monospace"
      font-size="8" fill="{t['subtitle']}" letter-spacing="2">
    {len(layers)} LAYERS · {_fmt(total_p)} PARAMETERS
  </text>''')

    # ── Dashed edge connectors (PlotNeuralNet style, drawn before blocks) ─────
    ds = f'stroke="{t["conn"]}" stroke-width="0.8" stroke-dasharray="3,2" fill="none"'
    for i in range(len(layers) - 1):
        h1, d1, fw1 = dims[i]
        h2, d2,  _  = dims[i + 1]

        # Right-face corners of block i (4 corners of the depth parallelogram)
        tr   = (xs[i] + fw1,             y_center - h1 / 2)
        br   = (xs[i] + fw1,             y_center + h1 / 2)
        tr_b = (xs[i] + fw1 + d1*_COS,  y_center - h1 / 2 - d1*_SIN)
        br_b = (xs[i] + fw1 + d1*_COS,  y_center + h1 / 2 - d1*_SIN)

        # Left-face corners of block i+1 — front AND back so each pair matches
        tl   = (xs[i + 1],              y_center - h2 / 2)
        bl   = (xs[i + 1],              y_center + h2 / 2)
        tl_b = (xs[i + 1] + d2*_COS,   y_center - h2 / 2 - d2*_SIN)
        bl_b = (xs[i + 1] + d2*_COS,   y_center + h2 / 2 - d2*_SIN)

        for x1, y1, x2, y2 in [
            (tr[0],   tr[1],   tl[0],   tl[1]),    # top-front  → top-front
            (br[0],   br[1],   bl[0],   bl[1]),    # bot-front  → bot-front
            (tr_b[0], tr_b[1], tl_b[0], tl_b[1]), # top-back   → top-back
            (br_b[0], br_b[1], bl_b[0], bl_b[1]), # bot-back   → bot-back
        ]:
            out.append(
                f'  <line x1="{x1:.1f}" y1="{y1:.1f}" '
                f'x2="{x2:.1f}" y2="{y2:.1f}" {ds}/>'
            )

    # ── Blocks ────────────────────────────────────────────────────────────────
    for (name, module), (h, d, fw), x in zip(layers, dims, xs):
        color = _col(type(module).__name__, pal)
        by    = y_center + h / 2
        out.append(f'  <!-- {type(module).__name__} ({name}) -->')
        out.append(_block(x, by, h, d, fw, color))

    # ── Labels (all at same y, below the deepest block bottom) ────────────────
    ly = y_center + max_below + 15
    for (name, module), (h, d, fw), x in zip(layers, dims, xs):
        tn    = type(module).__name__
        lx    = x + (fw + d * _COS) / 2
        color = _col(tn, pal)
        ps    = _n_params(module)
        short = name if len(name) <= 10 else '…' + name[-9:]

        out.append(
            f'  <text x="{lx:.1f}" y="{ly}" text-anchor="middle" '
            f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
            f'font-size="8" font-weight="700" fill="{color}">{tn}</text>'
        )
        out.append(
            f'  <text x="{lx:.1f}" y="{ly+12}" text-anchor="middle" '
            f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
            f'font-size="7" fill="{t["lbl_attr"]}">{short}</text>'
        )
        if ps > 0:
            out.append(
                f'  <text x="{lx:.1f}" y="{ly+23}" text-anchor="middle" '
                f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
                f'font-size="7" fill="{t["lbl_params"]}">{_fmt(ps)}</text>'
            )

    out.append('</svg>')
    return '\n'.join(out)

# ── HTML wrapper ──────────────────────────────────────────────────────────────
def _make_html(svg, model_name, bg='#080c12'):
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{model_name} — prometheus</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ background:{bg}; width:100%; height:100%; overflow:hidden; }}
  #vp {{
    width:100vw; height:100vh; overflow:auto;
    display:flex; align-items:center; justify-content:flex-start;
    cursor:grab; user-select:none;
  }}
  #vp:active {{ cursor:grabbing; }}
  #inner {{ padding:40px 60px; flex-shrink:0; }}
  footer {{
    position:fixed; bottom:14px; left:50%; transform:translateX(-50%);
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    font-size:8px; color:#0d1b2a; letter-spacing:3px; pointer-events:none;
  }}
</style>
</head>
<body>
<div id="vp"><div id="inner">
{svg}
</div></div>
<footer>PROMETHEUS · MODEL VISUALISER</footer>
<script>
  const vp = document.getElementById('vp');
  let drag=false, ox,oy,sl,st;
  vp.addEventListener('mousedown', e=>{{
    drag=true; ox=e.clientX; oy=e.clientY;
    sl=vp.scrollLeft; st=vp.scrollTop;
  }});
  document.addEventListener('mouseup',   ()=>drag=false);
  document.addEventListener('mousemove', e=>{{
    if(!drag) return;
    vp.scrollLeft = sl-(e.clientX-ox);
    vp.scrollTop  = st-(e.clientY-oy);
  }});
  // Centre diagram on load
  window.addEventListener('load', ()=>{{
    const inner = document.getElementById('inner');
    const ow = inner.offsetWidth, oh = inner.offsetHeight;
    const vw = vp.offsetWidth,    vh = vp.offsetHeight;
    if(ow > vw) vp.scrollLeft = (ow-vw)/2;
    if(oh > vh) vp.scrollTop  = (oh-vh)/2;
  }});
</script>
</body>
</html>'''

# ── Public API ────────────────────────────────────────────────────────────────
def visualise(model, filename=None, open_browser=True, theme='default'):
    """
    Generate a 3D PlotNeuralNet-style diagram for any prometheus model.
    Works on any object storing Module instances as attributes.

    model        : model to visualise
    filename     : output HTML path; None → temp file
    open_browser : auto-open in browser (default True)
    theme        : colour theme — 'default' or 'mbdtf'
    Returns      : path to generated HTML file
    """
    model_name = type(model).__name__
    layers = _collect(model)

    if not layers:
        print(f"[visualise] No layers found on '{model_name}'. "
              "Store layers as instance attributes on your model.")
        return None

    t       = _THEMES.get(theme, _THEMES['default'])
    svg     = _make_svg(model_name, layers, theme=theme)
    content = _make_html(svg, model_name, bg=t['bg'])

    if filename is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix='.html', prefix='prometheus_viz_',
            delete=False, mode='w', encoding='utf-8')
        filename = tmp.name
        tmp.write(content)
        tmp.close()
    else:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f'[visualise] {model_name} | {len(layers)} layers | '
          f'{_fmt(sum(_n_params(m) for _,m in layers))} params')
    print(f'[visualise] Saved -> {os.path.abspath(filename)}')
    if open_browser:
        webbrowser.open(f'file:///{os.path.abspath(filename)}')
    return filename
