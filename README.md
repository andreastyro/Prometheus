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
| GPT 4×256, bf16 | **2.4** | 7.8 | 2.2 |
| GPT 12×768, TF32 | **22.9** | 27.4 | 26.0 |
| GPT 12×768, bf16 | **17.9** | 26.0 | 17.6 |
| GPT 12×768, float32 | **77.2** | 77.3 | 80.0 |
| CNN, batch 64 | **1.0** | 1.6 | 1.8 |
| `p.fuse` swish + MSE | **0.40** | 0.59 | 0.54 |
| LSTM / GRU (with a CUDA Graph) | **2.2 / 2.0** | 2.3 / 2.4 | |
| RNN | 2.0 | 0.9 | 1.0 |

Behind: the plain RNN (cuDNN's persistent kernel); bf16 is within 2% of `torch.compile`.
On a GTX 970 (PyTorch 2.14 eager): GPT 16.9 vs 18.9 ms, CNN 4.4 vs 5.2, RNN 2.3 vs 3.5, LSTM and GRU even.

On the CPU (Ryzen AI 7 350, MKL, against PyTorch 2.15 on the same machine; ms, forward +
backward, [tests/bench_ops.py](tests/bench_ops.py) and the `bench_gpu_*.py --device cpu` steps):

| | Prometheus | PyTorch |
|---|---|---|
| GPT 4×256 training step | **184-191** | 188-193 |
| CNN training step, batch 64 | 45.5 | 42.4 |
| GRU training step | 52.1 | 49.8 |
| Attention, 8×256×768, 12 heads | **147** | 165 |
| GELU, 1M floats (forward) | **0.47** | 1.3-1.7 |
| Conv2D 16×32×32×32 → 64 | 11.7 | 9.3 |
| LayerNorm 8×256×768 | 2.7 | 1.2 |
| LSTM 64×32, 128 → 256 | 34.5 | 25.7 |

Element-wise ops run at PyTorch's speed, with GELU, the LSTM gates and float32 ↔ half
conversions in AVX2 / AVX-512 picked at runtime. Conv2D and LSTM trail PyTorch's oneDNN
kernels. `matmul` uses MKL or OpenBLAS when one loads (`p.backend()`, `p.set_blas`), else a
built-in packed AVX2 kernel within reach of MKL on large products (2048³: 310-320 GFLOP/s
against MKL's 320-440 across runs; 1024×768×3072: 280 vs 310-320,
[tests/bench_gemm.py](tests/bench_gemm.py)). A CI job fails if the CPU falls behind PyTorch
past a set ratio.

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

## Configuration

| Setting | What it picks |
|---|---|
| `p.set_blas("mkl" \| "openblas" \| "tiled" \| "auto")`, `PROMETHEUS_BLAS` | CPU matmul: a BLAS library loaded at runtime, or the built-in packed kernel (`p.backend()` shows the choice) |
| `PROMETHEUS_SIMD=base \| avx2 \| avx512` | CPU vector kernels; AVX2 by default where the CPU has it |
| `p.set_float32_matmul_precision("high")` | TF32 tensor cores for float32 GPU matmuls |
| `p.set_attention_backend("auto" \| "math" \| "flash")`, `PROMETHEUS_ATTENTION` | GPU attention |
| `p.set_default_autocast(p.bfloat16)` | GPU mixed precision without an `autocast` scope |
| `PROMETHEUS_ALLOCATOR=async` | the driver's stream-ordered GPU allocator instead of the caching one |
| `PROMETHEUS_CUDNN=0`, `PROMETHEUS_KERNEL_CACHE` | turn cuDNN off; where compiled GPU kernels are kept |

Tested: CPython 3.12 and 3.14; Windows (MSVC 2022, MinGW GCC 15), Linux
(GCC, Ubuntu in CI), macOS on Apple Silicon (Clang, in CI); x86-64 with AVX2 /
AVX-512 picked at runtime, ARM64 on the portable code. GPU: NVIDIA, Maxwell (GTX 900) onward with CUDA 12's
NVRTC and cuBLAS, Turing onward with CUDA 13; bfloat16 and TF32 on compute
8.0+. Tested on a GTX 970, T4 and A100.

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

- The CPU conv and LSTM gaps to oneDNN (a direct-convolution kernel, or oneDNN bundled)
- bf16 gradients end to end on the GPU, if bf16 becomes the default
- Meta-learning and continual-learning methods on top of higher-order gradients
