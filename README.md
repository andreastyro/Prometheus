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

ids    = p.Tensor([batch, seq_len], input_ids, p.int64)   # token ids, exact as int64
logits = model.forward(ids)                            # [batch, seq_len, vocab_size]
loss   = p.cross_entropy_sparse_seq(logits, target_ids)  # targets flattened, batch * seq_len

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
| **Tensor** | N-dimensional, strided storage with 64-bit shapes; `zeros`, `ones`, `full`, `randn`, `detach`, `half`/`to_float`, NumPy interop via `numpy()` / `from_numpy()` |
| **dtype / device** | `float32`, `float64`, `float16`, `bfloat16`, `int64`, `int32`, `bool`, converted with `to`; kernels compute in float32 and refuse other dtypes rather than misread them. Device is `cpu` (the field exists so a GPU is an addition, not a redesign) |
| **Views** | `transpose`, `permute`, `reshape`, `contiguous` — O(1), share storage, differentiable; every op accepts a view |
| **Autograd** | Reverse-mode AD, graph built dynamically on the forward pass, `backward()` via topological sort, per-tensor `requires_grad`; `grad` with create_graph for gradients of gradients (Hessian-vector products, MAML) |
| **Ops** | `add`, `subtract`, `multiply`, `divide` (numpy broadcasting), `matmul` (batched, broadcast batch dims), `sum`, `mean` (any dim, with keepdim), `pow_op`, `sqrt_op`, `abs_op`, `exp_op`, `log_op`, `clip`, `max_op`, `min_op`, `argmax`, `concat_op`, `linear_op` |
| **Activations** | `relu`, `gelu`, `sigmoid`, `tanh_op`, `softmax` |

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

- `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`, `r2_score`
- `save_model` / `load_model`, `save_checkpoint` / `load_checkpoint` (weights + epoch + best loss), written as safetensors -- every dtype, readable by PyTorch and numpy; files in the old format still load
- `save_safetensors` / `load_safetensors` for named tensors, including pretrained weights from the Hugging Face hub
- `train()` loop helper
- (C++) `DataLoader` — batching, per-epoch reshuffle; `data_split` for train/val/test
- (C++) `read_csv`, `load_image` (PNG/JPG → tensor), `EarlyStopping`, `model_summary`
- ONNX export via [prometheus_onnx.py](prometheus_onnx.py)

---

## Performance

`matmul` goes through a BLAS library chosen **at runtime**, not at build time.
OpenBLAS and Intel MKL both export the standard `cblas_sgemm`, so the module
loads whichever is selected and calls through it. One build serves every case,
and switching costs no rebuild:

```python
>>> import prometheus as p
>>> p.backend()
{'matmul': 'mkl', 'library': '...\\mkl_rt.3.dll', 'blas_threads': 8, 'threads': 16}
>>> p.set_blas("openblas")      # or "mkl", "tiled", "auto"
```

| Backend | Where it comes from |
|---|---|
| **Intel MKL** | `pip install mkl` (or `pip install prometheus[mkl]`) |
| **OpenBLAS** | system package, or `libopenblas` next to the module |
| **Tiled** | built in: cache-friendly tiling, parallel with OpenMP |

By default (`"auto"`) MKL is tried first, then OpenBLAS, then the built-in
kernel. `PROMETHEUS_BLAS=mkl|openblas|tiled` picks one at startup, and
`PROMETHEUS_BLAS_PATH` names the exact library file. Asking for a library that
cannot be loaded raises an error rather than quietly falling back: always
check `backend()` before trusting a benchmark.

Measured on a 16.8M-parameter GPT (vocab 16000, embed 384, 6 heads, 6 layers),
batch 8, sequence 128, one full training step including backward and Adam, on
a Ryzen AI 7 350. PyTorch 2.15 CPU was run on the identical architecture, in
the same session, interleaved with Prometheus:

| | tok/s |
|---|---|
| Prometheus, tiled + OpenMP | 218 |
| Prometheus, OpenBLAS | 1523 |
| Prometheus, MKL | 1532-1681 |
| PyTorch 2.15 (MKL) | 1917-1976 |

Absolute numbers on a laptop move with power and thermal state by as much as
40%, which is why the comparison is interleaved.

Per layer, forward + backward, against PyTorch 2.15 on the same shapes
(best of three interleaved rounds):

| Layer | Shape | Prometheus | PyTorch | |
|---|---|---|---|---|
| Conv2D | [16,32,32,32] -> 64, k3 | 9.13 ms | 2.91 ms | 3.1x slower |
| ConvTranspose2D | [16,64,16,16] -> 32, k4 s2 | 4.07 ms | 1.48 ms | 2.8x slower |
| MaxPool2D | [16,64,32,32], k2 | 0.72 ms | 1.03 ms | 1.4x faster |
| AvgPool2D | [16,64,32,32], k2 | 0.24 ms | 0.52 ms | 2.2x faster |
| BatchNorm | [16,64,32,32] | 0.81 ms | 1.38 ms | 1.7x faster |
| GroupNorm | [16,64,32,32], 8 groups | 0.44 ms | 0.65 ms | 1.5x faster |
| RNN | seq 64, batch 32, 128 -> 256 | 6.24 ms | 5.92 ms | 1.1x slower |
| LSTM | seq 64, batch 32, 128 -> 256 | 24.12 ms | 10.59 ms | 2.3x slower |
| GRU | seq 64, batch 32, 128 -> 256 | 15.98 ms | 18.82 ms | 1.2x faster |
| Embedding | [32,128] ids, 16000 x 384 | 0.89 ms | 2.82 ms | 3.2x faster |
| SGD (momentum) | 16.8M parameters | 5.60 ms | 8.07 ms | 1.4x faster |
| AdamW | 16.8M parameters | 8.06 ms | 22.89 ms | 2.8x faster |
| RMSprop | 16.8M parameters | 5.53 ms | 14.81 ms | 2.7x faster |

PyTorch runs its convolutions through oneDNN's direct-convolution kernels;
against its own im2col path (oneDNN off) the convolutions here are 1.3-1.6x. Benchmark with
[tests/benchmark_matmul.cpp](tests/benchmark_matmul.cpp).

---

## Install

### Python

```bash
pip install -e . --no-build-isolation
```

Parallelise across cores with OpenMP:

```bash
PROMETHEUS_USE_OPENMP=1 pip install -e . --no-build-isolation
pip install mkl                  # optional: fastest matmul, picked up automatically
```

The BLAS library is not a build option; see [Performance](#performance).

Builds are portable by default. `PROMETHEUS_NATIVE=1` tunes for the building
machine's CPU (`-march=native`, or `/arch:AVX2` with MSVC) — fine for a local
build, wrong for a wheel, which would crash on older CPUs.

The package bundles the pure-Python models alongside the compiled core:

```python
import prometheus as p                 # the C++ core
from prometheus.models import MLP, CNN, ResNet, UNet, VAE, GAN
from prometheus.gpt import GPT         # a GPT written in Python on the C++ ops
```

Prebuilt wheels for Windows, Linux, and macOS are produced by
[.github/workflows/build_wheels.yml](.github/workflows/build_wheels.yml).

### C++

Requires CMake and a C++17 compiler.

```bash
cmake -B build                     # tiled matmul, no external BLAS (default)
cmake --build build
ctest --test-dir build             # 25 test binaries
```

Each component has its own test executable (`test_tensor`, `test_gpt`, `test_attention`, …),
built alongside the library. `ctest` runs them all with the correct working
directory, which some tests need in order to find their fixtures.

#### OpenMP and BLAS

`cmake -B build -DUSE_OPENMP=ON` parallelises across cores. BLAS needs no
build flag: the test binaries load MKL or OpenBLAS at runtime if one is on the
library path or beside the executable, and `PROMETHEUS_BLAS=tiled ctest`
forces the built-in kernel.

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
| [maml_sine.py](examples/maml_sine.py) | MAML few-shot sine regression: differentiates through the inner step with `p.grad(..., create_graph=True)`. After one step on 10 points, MSE 1.69 against 4.50 for a random init after ten |

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
