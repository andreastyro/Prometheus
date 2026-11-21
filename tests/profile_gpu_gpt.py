"""Where a GPT training step spends its GPU time, kernel by kernel: Prometheus
(its event-based kernel profiler) or PyTorch (torch.profiler), the same model
as tests/bench_gpu_gpt.py.

    python tests/profile_gpu_gpt.py prometheus|torch [--top N] [--steps N]
        [--amp bf16] [--compile] [--manual-attn]

PyTorch's attention is scaled_dot_product_attention unless --manual-attn;
--compile runs it through torch.compile (max-autotune).
"""
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("lib", choices=["prometheus", "torch"])
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--vocab", type=int, default=2048)
ap.add_argument("--seq", type=int, default=128)
ap.add_argument("--embed", type=int, default=256)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--amp", choices=["bf16"], help="autocast to bfloat16")
ap.add_argument("--compile", action="store_true", help="PyTorch: torch.compile(mode='max-autotune')")
ap.add_argument("--manual-attn", action="store_true", help="PyTorch: matmul / mask / softmax attention")
args = ap.parse_args()

import numpy as np

rng = np.random.default_rng(0)
ids = rng.integers(0, args.vocab, (args.batch, args.seq))
targets = rng.integers(0, args.vocab, args.batch * args.seq)

if args.lib == "prometheus":
    import prometheus as p
    p.Tensor.manual_seed(0)
    model = p.GPT(args.vocab, args.seq, args.embed, args.heads, args.layers)
    model.cuda()
    opt = p.AdamW(model.parameters(), 3e-4)
    x = p.Tensor([args.batch, args.seq], [float(v) for v in ids.reshape(-1)], p.int64).cuda()
    # targets as an int64 tensor on the device, built once: the list form
    # uploads them every step
    tg = p.Tensor([len(targets)], [float(t) for t in targets], p.int64)
    tg = tg.cuda()

    def step():
        opt.zero_grad()
        if args.amp:
            with p.autocast(p.bfloat16):
                loss = p.cross_entropy_sparse_seq(model.forward(x), tg)
        else:
            loss = p.cross_entropy_sparse_seq(model.forward(x), tg)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    p.cuda_synchronize()
    p.cuda_profile(True)
    p.cuda_profile_reset()
    for _ in range(args.steps):
        step()
    rows = p.cuda_profile_entries()
    p.cuda_profile(False)
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.profiler import profile, ProfilerActivity
    torch.manual_seed(0)
    E, H, L, V, S = args.embed, args.heads, args.layers, args.vocab, args.seq

    # the same model as tests/bench_gpu_gpt.py: attention written out, output tied to the embedding
    class Block(nn.Module):
        def __init__(self, e, h):
            super().__init__()
            self.h = h
            self.n1, self.n2 = nn.LayerNorm(e), nn.LayerNorm(e)
            self.q, self.k, self.v = (nn.Linear(e, e, bias=False) for _ in range(3))
            self.o = nn.Linear(e, e, bias=False)
            self.f1, self.f2 = nn.Linear(e, 4 * e), nn.Linear(4 * e, e)

        def forward(self, x):
            b, s, e = x.shape
            y = self.n1(x)
            q, k, v = (m(y).view(b, s, self.h, e // self.h).transpose(1, 2) for m in (self.q, self.k, self.v))
            if args.manual_attn:
                att = (q @ k.transpose(-2, -1)) / (e // self.h) ** 0.5
                att = att.masked_fill(torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), 1),
                                      float("-inf"))
                y = F.softmax(att, -1) @ v
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).reshape(b, s, e)
            x = x + self.o(y)
            return x + self.f2(F.gelu(self.f1(self.n2(x)), approximate="tanh"))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(V, E)
            self.pos = nn.Embedding(S, E)
            # initialised as Prometheus's GPT, N(0, 0.02): nn.Embedding's N(0, 1)
            # with the output tied to it started the loss at ~34 instead of ~log(vocab)
            nn.init.normal_(self.tok.weight, 0.0, 0.02)
            nn.init.normal_(self.pos.weight, 0.0, 0.02)
            self.blocks = nn.ModuleList(Block(E, H) for _ in range(L))
            self.ln = nn.LayerNorm(E)

        def forward(self, i):
            x = self.tok(i) + self.pos(torch.arange(i.shape[1], device=i.device))
            for b in self.blocks:
                x = b(x)
            return self.ln(x) @ self.tok.weight.t()

    dev = torch.device("cuda")
    model = GPT().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    x = torch.tensor(ids, device=dev)
    tg = torch.tensor(targets, device=dev)

    fwd = torch.compile(model, mode="max-autotune") if args.compile else model

    def step():
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
            loss = F.cross_entropy(fwd(x).view(-1, V).float(), tg)
        loss.backward()
        opt.step()

    for _ in range(5 if args.compile else 3):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
        if t > 0 and e.device_type.name == "CUDA":
            rows.append((e.key, t / 1000.0, e.count))
    rows.sort(key=lambda r: -r[1])

total = sum(r[1] for r in rows)
print("%s: GPU time per step %.2f ms over %d kinds" % (args.lib, total / args.steps, len(rows)))
for name, ms, calls in rows[:args.top]:
    print("  %-60s %8.3f ms/step %5.1f%% %6.1f calls/step"
          % (name[:60], ms / args.steps, 100 * ms / total, calls / args.steps))
