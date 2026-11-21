<p align="center">
  <img src="assets/logo.svg" alt="Prometheus — machine learning, unbound" width="620">
</p>

<p align="center">
  A deep learning framework written from scratch in C++17, with Python bindings.<br>
  No PyTorch, no TensorFlow, no Eigen — tensors, autograd, layers, optimizers and GPU kernels written by hand.
</p>

---

## What it is

Reverse-mode autograd over a dynamic graph, a layer library up to a GPT with RoPE and a KV cache,
and a GPU backend that loads the NVIDIA driver at runtime — one build for machines with and without
a GPU. The only build dependency is pybind11; BLAS, cuBLAS and cuDNN are optional and found at runtime.

```python
import prometheus as p

model = p.Sequential([p.Linear(2, 8), p.ReLU(), p.Linear(8, 1), p.Sigmoid()])
opt = p.Adam(model.parameters(), lr=0.01)
for epoch in range(1000):
    loss = p.mse_loss(model.forward(X), y)
    loss.backward()
    opt.step()
    opt.zero_grad()
```

## Training a GPT

```python
model = p.GPT(vocab_size, max_seq_len, embed_dim=64, num_heads=4, num_layers=2, rope=True)
opt   = p.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
ids   = p.Tensor([batch, seq_len], input_ids, p.int64)
loss  = p.cross_entropy_sparse_seq(model.forward(ids), target_ids)
loss.backward()
p.clip_grad_norm(model.parameters(), 1.0)
opt.step()

cache  = model.make_kv_cache()                 # generation, O(n) per token
logits = model.forward_cached(token_id, cache)
```

`python examples/gpt_shakespeare.py [text.txt]` trains one end to end.

---

## Implemented

| | |
|---|---|
| **Tensors** | strided N-d tensors; `float32`, `float64`, `float16`, `bfloat16`, `int64`, `int32`, `bool` with PyTorch's promotion rules; views (`transpose`, `permute`, `reshape`); NumPy interop |
| **Autograd** | dynamic graph, `backward()`, `grad(..., create_graph=True)` for higher-order gradients |
| **Ops** | arithmetic with broadcasting, `matmul`, reductions, `softmax`, activations, comparisons, `where`, `masked_fill`, `concat_op` |
| **Layers** | `Linear`, `Dropout`, `BatchNorm`, `GroupNorm`, `Conv2D`, `ConvTranspose2D`, `MaxPool2D`, `AvgPool2D`, `RNN`, `LSTM`, `GRU` (`reset_after=True` is PyTorch's form), `Embedding`, `LayerNorm`, `MultiHeadAttention`, `TransformerBlock`, `GPT` |
| **Losses** | MSE, MAE, Huber, BCE, cross-entropy (dense, sparse, per sequence), KL, contrastive, L1 / L2 |
| **Training** | `SGD`, `Adam`, `AdamW`, `RMSprop`; LR schedules; `clip_grad_norm`; `autocast` and `GradScaler` |
| **I/O** | safetensors save / load (PyTorch-compatible, Hugging Face weights), checkpoints, ONNX export, metrics |

Initialisation matches PyTorch's defaults.

---

## GPU

```bash
pip install nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12      # nvidia-cudnn-cu12 optional
```

```python
model.cuda()
ids = ids.cuda()
loss = p.cross_entropy_sparse_seq(model.forward(ids), targets)
```

Kernels are compiled for the GPU present (Maxwell onward) and cached on disk. Every layer, loss and
optimizer runs on the GPU, tested against the CPU.

| Feature | |
|---|---|
| Mixed precision | `autocast(p.bfloat16)` keeps the whole path in bf16; `p.set_default_autocast` turns it on without a scope |
| TF32 | `p.set_float32_matmul_precision("high")` |
| CUDA Graphs | `p.CUDAGraph`, or `@p.compile` to capture and replay a whole step |
| Fusion | `@p.fuse` compiles your own element-wise functions, losses or whole blocks (`p.fx` ops) into generated kernels, with gradients to any order (`prometheus.compiler`) |
| Attention | `p.set_attention_backend("auto" \| "math" \| "flash")`; flash keeps memory linear in the sequence length |
| Data loading | `p.prefetch(batches)` uploads the next batch on a second stream |
| Diagnostics | `p.cuda_profile()`, `p.cuda_launch_counts()`, `p.cuda_memory_stats()` |

```python
@p.fuse
def swish_mse(x, t):
    y = x * p.fx.sigmoid(x) - t
    return p.fx.mean(y * y)
```

## Performance

One training step, ms, against PyTorch 2.15 (with flash attention) on an A100, run on
[Modal](https://modal.com) by [tests/modal_gpu.py](tests/modal_gpu.py):

| Model | Prometheus | PyTorch eager | torch.compile |
|---|---|---|---|
| GPT 4×256, float32 | **5.1** | 7.2 | 4.8 |
| GPT 4×256, bf16 | **2.6** | 7.7 | 2.4 |
| GPT 12×768, TF32 | **23.3** | 25.2 | 25.2 |
| GPT 12×768, bf16 | **18.9** | 25.9 | 17.6 |
| GPT 12×768, float32 | 78.2 | 78.0 | 80.4 |
| CNN, batch 64 | **1.0** | 1.6 | 1.8 |
| `p.fuse` swish + MSE | **0.40** | 0.59 | 0.54 |
| LSTM / GRU (with a CUDA Graph) | **2.2 / 2.0** | 2.3 / 2.4 | |
| RNN | 2.0 | 0.9 | 1.0 |

Behind: bf16 against `torch.compile` (it fuses across ops), and the plain RNN (cuDNN's persistent
kernel). On a GTX 970 the GPT step is 16.5 ms against PyTorch's 19.4.

On the CPU, `matmul` uses MKL, OpenBLAS or a built-in tiled kernel, chosen at runtime
(`p.backend()`, `p.set_blas`). A 16.8M-parameter GPT step runs at 1530-1680 tokens/s with MKL
against PyTorch's 1920-1980 on a Ryzen AI 7 350; pooling, norms, GRU, embeddings and optimizers
are faster than PyTorch's, convolutions about 3x slower.

## Install

```bash
PROMETHEUS_USE_OPENMP=1 pip install . --no-build-isolation
pip install mkl                       # optional: fastest CPU matmul
```

C++: `cmake -B build && cmake --build build && ctest --test-dir build`.
`PROMETHEUS_NATIVE=1` tunes for the building CPU. Wheels come from
[build_wheels.yml](.github/workflows/build_wheels.yml).

GPU tests: [tests/test_cuda.cpp](tests/test_cuda.cpp) and [tests/test_cuda.py](tests/test_cuda.py)
(they skip without a GPU); `python tests/build_gpu_tests.py --run` builds and runs the C++ suite;
`modal run tests/modal_gpu.py --gpu A100` runs everything plus the benchmarks on a rented GPU.

## Examples

| File | What it shows |
|---|---|
| [xor_example.py](examples/xor_example.py) | Smallest training loop |
| [mnist_conv.py](examples/mnist_conv.py) | MNIST with `Conv2D` |
| [sentiment_rnn.py](examples/sentiment_rnn.py) | Sequence classification |
| [lm_tiktoken.py](examples/lm_tiktoken.py) | Token-level language model |
| [gpt_shakespeare.py](examples/gpt_shakespeare.py) | Full GPT: RoPE, AdamW, cosine schedule, KV cache |
| [maml_sine.py](examples/maml_sine.py) | MAML through `p.grad(..., create_graph=True)` |

## Layout

```
include/ml/   headers: tensor, ops, autograd, nn/, optim/, data/, utils/, cuda
src/          implementations; src/cuda/ the GPU kernels and runtime
python/       bindings, prometheus.compiler (IR, codegen, executor), fuse
tests/        one test per component, GPU suites, benchmarks
examples/     runnable Python examples
```

## Roadmap

- Close the bf16 gap to `torch.compile` (tensor-core flash attention)
- Faster recurrent layers and CPU convolutions
- Meta-learning and continual-learning methods on top of higher-order gradients
