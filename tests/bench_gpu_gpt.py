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
ap.add_argument("--graph", action="store_true", help="Prometheus: capture the step in a CUDA Graph and time replays")
ap.add_argument("--compile", action="store_true", help="PyTorch only: torch.compile the model (Inductor; Triton needs compute 7.0+)")
ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
ap.add_argument("--rope", action="store_true", help="RoPE instead of learned positions (Prometheus only; PyTorch keeps its learned positions)")
ap.add_argument("--clip", action="store_true", help="clip_grad_norm(1.0) every step, as real training loops do")
ap.add_argument("--tf32", action="store_true", help="float32 matmuls on TF32 tensor cores (set_float32_matmul_precision('high'))")
ap.add_argument("--amp", choices=["bf16", "fp16"], help="mixed precision: autocast to bf16 / fp16 (fp16 with a GradScaler)")
ap.add_argument("--compile-mode", default="default", help="torch.compile mode: default, max-autotune, reduce-overhead")
ap.add_argument("--clip-norm", action="store_true", help="with --clip, read the norm back (Prometheus; a sync each step)")
ap.add_argument("--issue", action="store_true", help="also time how long the CPU takes to issue a step (before waiting for the GPU)")
args = ap.parse_args()

import numpy as np

rng = np.random.default_rng(0)
ids = rng.integers(0, args.vocab, (args.batch, args.seq))
targets = rng.integers(0, args.vocab, args.batch * args.seq)
tokens_per_step = args.batch * args.seq

if args.lib == "prometheus":
    import prometheus as p
    print("prometheus:", p.cuda_status() if args.device == "cuda" else p.backend())
    p.Tensor.manual_seed(0)
    if args.tf32:
        p.set_float32_matmul_precision("high")
    model = p.GPT(args.vocab, args.seq, args.embed, args.heads, args.layers, rope=args.rope)
    if args.device == "cuda":
        model.cuda()
    opt = p.AdamW(model.parameters(), 3e-4)
    x = p.Tensor([args.batch, args.seq], [float(v) for v in ids.reshape(-1)], p.int64)
    if args.device == "cuda":
        x = x.cuda()
    # targets as an int64 tensor on the device, built once: the list form
    # uploads them every step
    tg = p.Tensor([len(targets)], [float(t) for t in targets], p.int64)
    if args.device == "cuda":
        tg = tg.cuda()

    amp = {"bf16": p.bfloat16, "fp16": p.float16}.get(args.amp)
    scaler = p.GradScaler() if args.amp == "fp16" else None

    def step():
        opt.zero_grad()
        if amp is not None:
            with p.autocast(amp):
                loss = p.cross_entropy_sparse_seq(model.forward(x), tg)
        else:
            loss = p.cross_entropy_sparse_seq(model.forward(x), tg)
        if scaler is not None:
            scaler.scale_loss(loss).backward()
            ok = scaler.unscale(model.parameters())
            if ok:
                opt.step()
            scaler.update(not ok)
            return loss
        loss.backward()
        if args.clip:
            p.clip_grad_norm(model.parameters(), 1.0, return_norm=args.clip_norm)
        opt.step()
        return loss

    def sync():
        if args.device == "cuda":
            p.cuda_synchronize()
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = torch.device(args.device)
    print("torch:", torch.__version__, torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu, %d threads" % torch.get_num_threads())
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
            # initialised as Prometheus's GPT, N(0, 0.02): nn.Embedding's N(0, 1)
            # with the output tied to it started the loss at ~34 instead of ~log(vocab)
            nn.init.normal_(self.tok.weight, 0.0, 0.02)
            nn.init.normal_(self.pos.weight, 0.0, 0.02)
            self.blocks = nn.ModuleList(Block(args.embed, args.heads) for _ in range(args.layers))
            self.ln = nn.LayerNorm(args.embed)

        def forward(self, i):
            x = self.tok(i) + self.pos(torch.arange(i.shape[1], device=i.device))
            for b in self.blocks:
                x = b(x)
            return self.ln(x) @ self.tok.weight.t()

    if args.tf32:
        torch.set_float32_matmul_precision("high")
    model = GPT().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    if args.compile:      # compiled during the warm-up steps, before timing
        model = torch.compile(model, mode=None if args.compile_mode == "default" else args.compile_mode)
    x = torch.tensor(ids, device=dev)
    tg = torch.tensor(targets, device=dev)

    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    scaler = torch.amp.GradScaler("cuda") if args.amp == "fp16" else None

    def step():
        opt.zero_grad()
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            loss = F.cross_entropy(model(x).view(-1, args.vocab).float(), tg)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            return loss
        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

for _ in range(3):          # warm-up: kernel compilation, allocator, cuBLAS heuristics
    step()
sync()
if args.graph and args.lib == "prometheus":
    # one captured step, replayed: the host work and the launches go
    graph = p.CUDAGraph()
    with graph.capture():
        captured_loss = step()

    def step():
        graph.replay()
        return captured_loss
    step()
    sync()
times, issue = [], []
for _ in range(args.steps):
    t0 = time.perf_counter()
    loss = step()
    issue.append(time.perf_counter() - t0)    # the CPU's part: the GPU may still be running
    sync()
    times.append(time.perf_counter() - t0)
if args.issue:
    print("CPU issue time: median %.1f ms of the step (the rest the CPU waits for the GPU)"
          % (sorted(issue)[len(issue) // 2] * 1e3))
if args.lib == "prometheus" and args.device == "cuda" and hasattr(p, "cuda_launch_counts"):
    # what one step issues: our kernels, cuBLAS / cuDNN calls, copies
    p.cuda_reset_launch_counts()
    step()
    sync()
    c = p.cuda_launch_counts()
    print("launches per step: %d kernels, %d cuBLAS / cuDNN calls, %d copies"
          % (c["kernels"], c["library"], c["copies"]))
best, med = min(times), sorted(times)[len(times) // 2]
print("%-10s %-4s batch %d x seq %d, embed %d, %d layers: median %.1f ms/step, best %.1f ms -> %.0f tokens/s (loss %.3f)"
      % (args.lib, args.device, args.batch, args.seq, args.embed, args.layers, med * 1e3, best * 1e3,
         tokens_per_step / med, float(loss.item())))
