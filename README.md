<p align="center">
  <img src="assets/logo.svg" alt="Prometheus — machine learning, unbound" width="620">
</p>

<p align="center">
  A deep learning framework written from scratch in C++17, with Python bindings.<br>
  No PyTorch, no TensorFlow, no Eigen — tensors, autograd, layers, and optimizers all implemented by hand.
</p>

---

## What it is

Prometheus is a working deep learning framework, not a teaching toy. It implements reverse-mode
automatic differentiation over a dynamically-built computation graph, and on top of that a layer
library that reaches all the way to a GPT-2-style decoder-only transformer with weight tying,
rotary position embeddings, and KV-cached inference.

The only external dependencies are pybind11 (for the Python module) and an optional BLAS backend
(OpenBLAS or Intel MKL) for matmul. Everything else — every gradient, every kernel, every weight
update — is in this repository.

```python
import prometheus as p

model = p.Sequential([
    p.Linear(2, 8),
    p.ReLU(),
    p.Linear(8, 1),
    p.Sigmoid(),
])

optimizer = p.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    out  = model.forward(X)
    loss = p.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## Training a GPT

The transformer stack is complete enough to train a small language model end to end:

```python
import tiktoken
import prometheus as p

model = p.GPT(vocab_size, max_seq_len, embed_dim=64, num_heads=4, num_layers=2, rope=True)
opt   = p.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
sched = p.CosineAnnealingLR(base_lr=3e-3, T_max=200, min_lr=3e-4)

ids    = p.Tensor([seq_len], [float(t) for t in input_ids])
logits = model.forward(ids)                            # [seq_len, vocab_size]
loss   = p.cross_entropy_sparse_seq(logits, target_ids)

loss.backward()
p.clip_grad_norm(model.parameters(), 1.0)
opt.step()
opt.zero_grad()
opt.lr = sched.step()

# Autoregressive generation with a KV cache — O(n) per token instead of O(n²)
cache = model.make_kv_cache()
logits = model.forward_cached(token_id, cache)         # [vocab_size]
```

Run the full example:

```bash
python examples/gpt_shakespeare.py                  # built-in Shakespeare snippet
python examples/gpt_shakespeare.py path/to/text.txt  # your own corpus
```

---

## Implemented

### Core

| | |
|---|---|
| **Tensor** | N-dimensional, flat storage; `zeros`, `ones`, `randn`, `reshape`, `transpose`, `detach`, `half`/`to_float`, NumPy interop via `numpy()` / `from_numpy()` |
| **Autograd** | Reverse-mode AD, graph built dynamically on the forward pass, `backward()` via topological sort, per-tensor `requires_grad` |
| **Ops** | `add`, `subtract`, `multiply`, `divide`, `matmul`, `pow`, `sqrt`, `abs`, `exp`, `log`, `clip`, `sum`, `mean`, `max`, `min`, `argmax`, `broadcast_add`, `reshape_op` |
| **Activations** | `relu`, `gelu`, `sigmoid`, `tanh`, `softmax` |

### Layers

| Group | Layers |
|---|---|
| **Core** | `Linear`, `Dropout`, `Flatten`, `Sequential`, `BatchNorm`, `GroupNorm` |
| **Vision** | `Conv2D`, `ConvTranspose2D`, `MaxPool2D`, `AvgPool2D` |
| **Recurrent** | `RNN`, `LSTM`, `GRU` (all with `forward_with_state`) |
| **NLP** | `Embedding`, `LayerNorm`, `PositionalEncoding`, `MultiHeadAttention` (causal + RoPE), `TransformerBlock`, `GPT` |

### Losses

`mse_loss` · `mae_loss` · `huber_loss` · `bce_loss` · `cross_entropy_loss` ·
`cross_entropy_sparse` · `cross_entropy_sparse_seq` · `kl_divergence` ·
`reconstruction_loss` · `contrastive_loss` · `l1_regularization` · `l2_regularization`

### Optimizers & schedules

`SGD` (with momentum) · `Adam` · `AdamW` (decoupled weight decay) · `RMSprop`
`StepLR` · `ExponentialLR` · `CosineAnnealingLR`
`clip_grad_norm` · `GradScaler` (mixed-precision loss scaling with overflow detection)

Adam and AdamW persist their momentum buffers through `save_state()` / `load_state()`.

### Data, metrics, utilities

- `DataLoader` — batching, per-epoch reshuffle; `data_split` for train/val/test
- `read_csv`, `load_image` (PNG/JPG → tensor)
- `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`, `r2_score`
- `save_model` / `load_model`, `save_checkpoint` / `load_checkpoint` (weights + epoch + best loss)
- `EarlyStopping`, `model_summary`, `train()` loop helper
- ONNX export via [prometheus_onnx.py](prometheus_onnx.py)

---

## Performance

`matmul` dispatches through a pluggable backend selected at build time:

| Backend | Notes |
|---|---|
| **OpenBLAS** | `cblas_sgemm`; best on AMD |
| **Intel MKL** | `cblas_sgemm`; best on Intel |
| **Tiled + OpenMP** | Cache-friendly `ikj` tiling, parallel outer loop |
| **Tiled** | Fallback, no external dependency |

CMake auto-detects what's available and falls back cleanly. Benchmark with
[tests/benchmark_matmul.cpp](tests/benchmark_matmul.cpp).

---

## Install

### Python

```bash
pip install -e . --no-build-isolation
```

Backend selection via environment variables before building:

```bash
PROMETHEUS_USE_OPENBLAS=1 pip install -e . --no-build-isolation   # OpenBLAS
PROMETHEUS_USE_MKL=1      pip install -e . --no-build-isolation   # Intel MKL
PROMETHEUS_USE_OPENMP=1   pip install -e . --no-build-isolation   # OpenMP
```

Prebuilt wheels for Windows, Linux, and macOS are produced by
[.github/workflows/build_wheels.yml](.github/workflows/build_wheels.yml).

### C++

Requires CMake and a C++17 compiler.

```bash
cmake -B build -DUSE_OPENBLAS=ON   # or -DUSE_MKL=ON / -DUSE_OPENMP=ON
cmake --build build
```

Each component has its own test executable (`test_tensor`, `test_gpt`, `test_attention`, …),
built alongside the library.

---

## Examples

| File | What it shows |
|---|---|
| [xor_example.py](examples/xor_example.py) | Smallest complete training loop |
| [regression.py](examples/regression.py) | Tabular regression |
| [binary_classifier.py](examples/binary_classifier.py) | Two-class classification |
| [mnist_linear.py](examples/mnist_linear.py) | MNIST with dense layers |
| [mnist_conv.py](examples/mnist_conv.py) | MNIST with `Conv2D` |
| [sentiment_rnn.py](examples/sentiment_rnn.py) | Sequence classification with an RNN |
| [text_generation.py](examples/text_generation.py) | Character-level language model |
| [lm_tiktoken.py](examples/lm_tiktoken.py) | Token-level LM on `cl100k_base` |
| [gpt_shakespeare.py](examples/gpt_shakespeare.py) | Full GPT: RoPE, AdamW, cosine schedule, KV-cache generation |

---

## Layout

```
include/ml/          headers
├── tensor.hpp  ops.hpp  loss.hpp  autograd.hpp  matmul_backend.hpp
├── nn/              module, linear, activations, dropout, flatten,
│   │                sequential, batchnorm, groupnorm
│   ├── vision/      conv2d, convtranspose2d, maxpool2d, avgpool2d
│   ├── rnn/         rnn, lstm, gru
│   └── nlp/         embedding, layernorm, positional_encoding,
│                    attention, transformer, gpt
├── optim/           optimizer, sgd, adam, adamw, rmsprop, scheduler, grad_scaler
├── data/            dataloader, csv, image
├── metrics/         metrics
└── utils/           model_io, trainer, summary, early_stopping, grad_clip

src/                 implementations, mirroring include/
python/bindings.cpp  pybind11 module
tests/               one test executable per component
examples/            runnable Python examples
```

---

## Roadmap

Tracked in detail in [plan.md](plan.md).

- **Meta-learning** — MAML, Reptile, ProtoNet; requires second-order gradients
- **Continual learning** — EWC, PackNet, ProgressiveNets; catastrophic-forgetting benchmarks
- **CUDA** — `device` field on Tensor, GPU kernels, cuBLAS matmul
