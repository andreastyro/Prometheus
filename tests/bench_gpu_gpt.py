"""GPT training step on the GPU: Prometheus against PyTorch, same model, same card.

    python tests/bench_gpu_gpt.py prometheus|torch [--batch B] [--seq S] ...

Run the two one after the other on the same machine; each prints tokens/s for
forward + backward + AdamW. The PyTorch model mirrors Prometheus's GPT: learned
token and position embeddings, pre-LayerNorm blocks with plain causal
attention (matmul, mask, softmax -- not a fused kernel), a 4x GELU MLP, a final
LayerNorm and an output projection tied to the token embedding.
"""
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("lib", choices=["prometheus", "torch"])
ap.add_argument("--vocab", type=int, default=2048)
ap.add_argument("--seq", type=int, default=128)
ap.add_argument("--embed", type=int, default=256)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--steps", type=int, default=20)
args = ap.parse_args()

import numpy as np

rng = np.random.default_rng(0)
ids = rng.integers(0, args.vocab, (args.batch, args.seq))
targets = rng.integers(0, args.vocab, args.batch * args.seq)
tokens_per_step = args.batch * args.seq

if args.lib == "prometheus":
    import prometheus as p
    print("prometheus:", p.cuda_status())
    p.Tensor.manual_seed(0)
    model = p.GPT(args.vocab, args.seq, args.embed, args.heads, args.layers)
    model.cuda()
    opt = p.AdamW(model.parameters(), 3e-4)
    x = p.Tensor([args.batch, args.seq], [float(v) for v in ids.reshape(-1)], p.int64).cuda()
    tg = [int(t) for t in targets]

    def step():
        opt.zero_grad()
        loss = p.cross_entropy_sparse_seq(model.forward(x), tg)
        loss.backward()
        opt.step()
        return loss

    def sync():
        p.cuda_synchronize()
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = torch.device("cuda")
    print("torch:", torch.__version__, torch.cuda.get_device_name(0))
    torch.manual_seed(0)

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
            att = (q @ k.transpose(-2, -1)) / (e // self.h) ** 0.5
            att = att.masked_fill(torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), 1), float("-inf"))
            y = (F.softmax(att, -1) @ v).transpose(1, 2).reshape(b, s, e)
            x = x + self.o(y)
            return x + self.f2(F.gelu(self.f1(self.n2(x)), approximate="tanh"))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(args.vocab, args.embed)
            self.pos = nn.Embedding(args.seq, args.embed)
            self.blocks = nn.ModuleList(Block(args.embed, args.heads) for _ in range(args.layers))
            self.ln = nn.LayerNorm(args.embed)

        def forward(self, i):
            x = self.tok(i) + self.pos(torch.arange(i.shape[1], device=i.device))
            for b in self.blocks:
                x = b(x)
            return self.ln(x) @ self.tok.weight.t()

    model = GPT().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    x = torch.tensor(ids, device=dev)
    tg = torch.tensor(targets, device=dev)

    def step():
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, args.vocab), tg)
        loss.backward()
        opt.step()
        return loss

    def sync():
        torch.cuda.synchronize()

for _ in range(3):          # warm-up: kernel compilation, allocator, cuBLAS heuristics
    step()
sync()
times = []
for _ in range(args.steps):
    t0 = time.perf_counter()
    loss = step()
    sync()
    times.append(time.perf_counter() - t0)
best, med = min(times), sorted(times)[len(times) // 2]
print("%-10s batch %d x seq %d, embed %d, %d layers: median %.1f ms/step, best %.1f ms -> %.0f tokens/s (loss %.3f)"
      % (args.lib, args.batch, args.seq, args.embed, args.layers, med * 1e3, best * 1e3,
         tokens_per_step / med, float(loss.item())))
