"""GRU(reset_after=True) against PyTorch's nn.GRU with the same weights:
outputs, final state and every gradient, on the CPU and (if present) the GPU.
Needs torch; skips without it.

    python tests/check_gru_vs_pytorch.py

The mapping: our W_x* is [in, H] (PyTorch's weight_ih rows, transposed),
W_h* likewise from weight_hh; b_r = b_ir + b_hr, b_z = b_iz + b_hz,
b_n = b_in, b_hn = b_hn.
"""
import sys

import numpy as np

try:
    import torch
except ImportError:
    print("torch not installed: skipped")
    sys.exit(0)
import prometheus as p

S, B, I, H = 7, 3, 5, 6
torch.manual_seed(0)
ref = torch.nn.GRU(I, H)
x = torch.randn(S, B, I, requires_grad=True)
h0 = (torch.randn(1, B, H) * 0.5).requires_grad_(True)
w = torch.randn(S, B, H)
wh = torch.randn(B, H)

out_t, hn_t = ref(x, h0)
loss_t = (out_t * w).sum() + (hn_t[0] * wh).sum()
loss_t.backward()

Wih, Whh = ref.weight_ih_l0.detach(), ref.weight_hh_l0.detach()
bih, bhh = ref.bias_ih_l0.detach(), ref.bias_hh_l0.detach()
gate = lambda m, k: m[k * H:(k + 1) * H]


def ours(device):
    g = p.GRU(I, H, reset_after=True)
    vals = [gate(Wih, k).t() for k in range(3)] + [gate(Whh, k).t() for k in range(3)] + \
           [gate(bih, 0) + gate(bhh, 0), gate(bih, 1) + gate(bhh, 1), gate(bih, 2), gate(bhh, 2)]
    for t, v in zip(g.parameters(), vals):
        t.data = [float(f) for f in v.contiguous().view(-1)]
    xp = p.Tensor([S, B, I], [float(f) for f in x.detach().view(-1)])
    hp = p.Tensor([B, H], [float(f) for f in h0.detach().view(-1)])
    wp = p.Tensor([S, B, H], [float(f) for f in w.view(-1)])
    whp = p.Tensor([B, H], [float(f) for f in wh.view(-1)])
    if device == "cuda":
        g.cuda()
        xp, hp, wp, whp = xp.cuda(), hp.cuda(), wp.cuda(), whp.cuda()
    xp.requires_grad = True
    hp.requires_grad = True
    out, hn = g.forward_with_state(xp, hp)
    loss = p.add(p.sum(p.multiply(out, wp), -1), p.sum(p.multiply(hn, whp), -1))
    loss.backward()
    return g, out, hn, xp, hp


worst = 0.0
ok = True


def cmp(name, a, b, tol):
    global worst, ok
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    err = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))
    good = err <= tol
    ok = ok and good
    print("  %-36s rel err %.2e  %s" % (name, err, "ok" if good else "** FAIL **"))


devices = ["cpu"] + (["cuda"] if p.cuda_available() else [])
for dev in devices:
    print("%s:" % dev)
    g, out, hn, xp, hp = ours(dev)
    cmp("output", out.cpu().data, out_t.detach().numpy(), 1e-5)
    cmp("final state", hn.cpu().data, hn_t.detach().numpy(), 1e-5)
    cmp("d input", xp.grad, x.grad.numpy(), 1e-4)
    cmp("d h0", hp.grad, h0.grad[0].numpy(), 1e-4)
    params = g.parameters()
    tg = [gate(ref.weight_ih_l0.grad, k).t() for k in range(3)] + \
         [gate(ref.weight_hh_l0.grad, k).t() for k in range(3)] + \
         [gate(ref.bias_ih_l0.grad, 0), gate(ref.bias_ih_l0.grad, 1), gate(ref.bias_ih_l0.grad, 2),
          gate(ref.bias_hh_l0.grad, 2)]
    names = ["dW_xr", "dW_xz", "dW_xn", "dW_hr", "dW_hz", "dW_hn", "db_r", "db_z", "db_n", "db_hn"]
    for n, t, e in zip(names, params, tg):
        cmp(n, t.grad, e.numpy(), 1e-4)
    # b_ir and b_hr get the same gradient in PyTorch; ours is their shared one
    cmp("db_r == PyTorch db_hr", params[6].grad, gate(ref.bias_hh_l0.grad, 0).numpy(), 1e-4)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
