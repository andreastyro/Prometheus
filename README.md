# Prometheus

A machine learning library built from scratch in C++, with Python bindings (in progress).

Prometheus implements the core components of a modern deep learning framework — tensors, automatic differentiation, neural network layers, optimizers, and data utilities — without relying on any external ML dependencies.

---

## Why

Most ML libraries are black boxes. Prometheus is an attempt to understand what's actually happening under the hood — every operation, every gradient, every weight update written by hand.

---

## What's implemented

### Tensor
- N-dimensional tensor with flat data storage
- Element-wise ops: add, subtract, multiply, divide, pow, sqrt, abs, exp, log, clip
- Matrix ops: matmul, transpose, reshape
- Reductions: sum, mean, max, min along an axis
- Factory methods: zeros, ones, randn

### Autograd
- Reverse-mode automatic differentiation
- Computation graph built dynamically on forward pass
- `backward()` via topological sort
- `requires_grad` flag per tensor

### Neural Network Layers
- `Linear` — fully connected layer
- `ReLU`, `Sigmoid`, `Tanh`, `Softmax` — activations
- `Dropout` — regularization
- `Sequential` — chain layers together

### Loss Functions
- `mse_loss` — mean squared error
- `mae_loss` — mean absolute error
- `binary_cross_entropy`
- `cross_entropy_loss`

### Optimizers
- `SGD` with optional momentum
- `Adam`
- `RMSprop`

### Data Utilities
- `DataLoader` — batch iteration, shuffling, reshuffle per epoch
- `data_split` — train/val/test split with optional shuffle
- `read_csv` — load tabular data from CSV into tensors
- `load_image` — load PNG/JPG images as tensors `[C, W, H]`

### Metrics
- `accuracy` — binary and multi-class

### Model Utilities
- `save` / `load` — serialize model weights to binary file
- `train` — training loop helper

---

## Project structure

```
include/ml/
├── tensor.hpp
├── ops.hpp
├── loss.hpp
├── autograd.hpp
├── nn/          — module, linear, activations, dropout, sequential
├── optim/       — optimizer, sgd, adam, rmsprop
├── data/        — dataloader, csv, image loader
├── metrics/     — accuracy, precision, recall etc.
└── utils/       — model_io, trainer

src/             — implementations
tests/           — one test file per component
```

---

## Build

Requires CMake and a C++17 compiler.

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

---

## Roadmap

- Convolutional layers (Conv2D, MaxPool2D)
- Recurrent layers (RNN, LSTM, GRU)
- Attention and Transformers
- Python bindings via pybind11
- GPU support via CUDA
