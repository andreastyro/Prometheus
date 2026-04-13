# ML Library Plan

A machine learning library written from scratch in C++, with Python bindings via pybind11.

---

## Goal
Build a minimal but functional ML library that can train neural networks, then expose it to Python so it can be used like a mini PyTorch.

---

## Progress

### Phase 1 — Tensor (Core Data Structure)
- [x] `Tensor` class — flat data storage + shape
- [x] Constructor — creates zeroed tensor from shape
- [x] `get(row, col)` — read a value
- [x] `set(row, col, val)` — write a value
- [x] `numel()` — total element count
- [x] `fill(val)` — fill all elements with a value
- [x] `zeros(shape)` — static factory
- [x] `ones(shape)` — static factory
- [x] `print()` — debug output (declared, body in tensor.cpp)
- [x] Second constructor — create tensor from existing data
- [x] `randn(shape)` — random normal values (needed for weight init)
- [x] `transpose()` — flip rows and cols
- [x] `reshape(new_shape)` — change shape without changing data

---

### Phase 2 — Ops (Math Operations)
- [x] `add(a, b)` — element-wise addition
- [x] `multiply(a, b)` — element-wise multiplication
- [x] `matmul(a, b)` — matrix multiplication
- [x] `subtract(a, b)` — element-wise subtraction
- [x] `divide(a, b)` — element-wise division
- [x] `relu(a)` — max(0, x)
- [x] `sigmoid(a)` — 1 / (1 + e^-x)
- [x] `tanh(a)` — hyperbolic tangent
- [x] `softmax(a)` — convert logits to probabilities
- [x] `log(a)` — element-wise natural log
- [x] `exp(a)` — element-wise e^x
- [x] `pow(a, p)` — element-wise power
- [x] `sqrt(a)` — element-wise square root
- [x] `abs(a)` — element-wise absolute value
- [x] `sum(a, axis)` — sum along a dimension
- [x] `mean(a, axis)` — mean along a dimension
- [x] `max_op(a)` — max element (moved from Tensor)
- [x] `min_op(a)` — min element (moved from Tensor)
- [x] `clip(min, max)` — clamp values to a range
- [x] `broadcast_add(a, b)` — add tensors of different shapes (e.g. add bias to batch)

---

### Phase 3 — Core Neural Network Layers
- [x] `Module` — abstract base class, every layer inherits from this
- [x] `Linear` — fully connected layer: matmul(input, weights) + bias
- [x] `ReLU` — activation layer
- [x] `Sigmoid` — activation layer
- [x] `Tanh` — activation layer
- [x] `Softmax` — activation layer (used in output for classification)
- [x] `Dropout` — randomly zero out neurons during training (regularization)
- [x] `Sequential` — chain layers together: `Sequential({Linear, ReLU, Linear})`

---

### Phase 4 — Loss Functions
- [x] `mse_loss(pred, target)` — mean squared error (regression)
- [x] `mae_loss(pred, target)` — mean absolute error (regression)
- [x] `binary_cross_entropy(pred, target)` — binary classification
- [x] `cross_entropy_loss(logits, targets)` — multi-class classification
---

### Phase 5 — Optimizers
- [x] `Optimizer` — abstract base class
- [x] `SGD` — stochastic gradient descent
- [x] `SGD with momentum` — SGD with momentum term, faster convergence
- [x] `Adam` — adaptive moment estimation, most commonly used
- [x] `RMSprop` — another adaptive optimizer, good for RNNs
---

### Phase 6 — Autograd (Automatic Differentiation)
This is the hardest phase — without it you have to compute gradients by hand.
- [x] Add `grad` field to Tensor — stores the gradient
- [x] Add `requires_grad` flag — whether to track this tensor
- [x] `GradNode` — stores the op and its inputs so we can walk backwards
- [x] `backward()` — topological sort of the graph, call each grad function
- [x] Rewrite ops to attach grad functions when `requires_grad=true`

---

### Phase 7 — Data Utilities
- [x] `DataLoader` — load data in batches
- [x] `shuffle()` — shuffle dataset each epoch
- [x] CSV reader — load tabular data from a file
- [x] Simple image loader — load grayscale images as tensors
- [x] Train/val/test split

---

### Phase 8 — Model Utilities
- [x] `save(path)` — save model weights to a file
- [x] `load(path)` — load model weights from a file
- [x] Training loop helper — forward, loss, backward, step
- [ ] `accuracy(pred, target)` — compute accuracy metric
- [ ] Loss/accuracy history tracking
- [ ] `ModelSummary` — print layer names, shapes, and parameter count
- [ ] Weight initialization strategies — `xavier`, `kaiming`, `random normal`
- [ ] `EarlyStopping` — stop training when val loss stops improving

---

### Phase 9 — Metrics
- [ ] `accuracy(pred, target)`
- [ ] `precision(pred, target)`
- [ ] `recall(pred, target)`
- [ ] `f1_score(pred, target)`
- [ ] `confusion_matrix(pred, target)`
- [ ] `r2_score(pred, target)` — for regression

---

### Phase 10 — Convolutional & Recurrent Layers

#### Phase 10a — Convolutional Layers (for image data)
- [ ] `Conv2D` — 2D convolution: slides a filter across an image, extracts features
- [ ] `MaxPool2D` — takes the max value in each region, reduces spatial size
- [ ] `AvgPool2D` — takes the average value in each region, reduces spatial size
- [ ] `Flatten` — converts a 3D tensor (channels, height, width) to 1D for a Linear layer
- [ ] `Conv2D transpose` — upsampling, used in generative models

#### Phase 10b — Recurrent Layers (for sequences and text)
- [ ] `RNN` — basic recurrent layer, processes one timestep at a time
- [ ] `LSTM` — long short-term memory, handles long sequences without vanishing gradients
- [ ] `GRU` — gated recurrent unit, simpler alternative to LSTM, similar performance

---

### Phase 11 — Attention & Transformers
- [ ] `Embedding` — maps integer tokens to dense vectors (used in NLP and recommendation)
- [ ] `LayerNorm` — normalizes across features, used in transformers
- [ ] `GroupNorm` — normalizes in groups of channels, good for small batches
- [ ] `MultiHeadAttention` — core of transformer models, attends to different parts of input
- [ ] `TransformerBlock` — attention + feedforward + layer norm combined
- [ ] `PositionalEncoding` — injects position info into token embeddings

---

### Phase 12 — Extended Features
These were deferred from earlier phases — implement after Phase 11 to round out the core library.

#### Layers
- [ ] `BatchNorm` — normalize activations across a batch

#### Loss Functions
- [ ] `huber_loss(pred, target)` — robust regression loss, less sensitive to outliers
- [ ] `kl_divergence(p, q)` — difference between two probability distributions (used in VAEs)
- [ ] `reconstruction_loss(input, output)` — input vs output similarity for autoencoders
- [ ] `contrastive_loss(anchor, positive, negative)` — pulls similar samples together (self-supervised)

#### Optimizers
- [ ] `learning rate scheduler` — decay lr during training (step, cosine, exponential)
- [ ] `L1/L2 weight decay` — regularization built into optimizer

---

### Phase 13 — Python Bindings
- [ ] Install pybind11
- [ ] Wrap `Tensor` class — expose data, shape, get, set, print
- [ ] Wrap ops — add, matmul, relu, sigmoid etc.
- [ ] Wrap layers — Linear, Sequential, Conv2D etc.
- [ ] Wrap optimizers — SGD, Adam
- [ ] Wrap loss functions
- [ ] Wrap metrics
- [ ] `setup.py` for pip install
- [ ] NumPy interop — convert between Tensor and np.array

---

### Phase 14 — Examples
- [ ] XOR problem — simplest possible neural network
- [ ] MNIST digit classifier — image classification with Linear layers
- [ ] MNIST with Conv2D — same task using convolutional layers
- [ ] House price regression — tabular data
- [ ] Binary classifier — two class problem
- [ ] Sentiment classification — simple NLP with RNN or LSTM
- [ ] Text generation — character-level language model

---

### Phase 15 — Advanced (Ambitious)
- [ ] ONNX export — save model in standard format other frameworks can load
- [ ] Mixed precision training — use float16 for speed

---

### Phase 16 — Meta-Learning
Train models that can adapt to new tasks with very few examples. Requires autograd (Phase 6) to be complete first since meta-learning involves differentiating through the training process itself.

#### Algorithms
- [ ] `MAML` — Model-Agnostic Meta-Learning, learns a good weight initialization that can be fine-tuned in a few steps
- [ ] `Reptile` — simpler alternative to MAML, easier to implement
- [ ] `ProtoNet` — Prototypical Networks, learns an embedding space for few-shot classification

#### Supporting infrastructure
- [ ] `TaskSampler` — samples N-way K-shot tasks from a dataset (e.g. 5 classes, 1 example each)
- [ ] `MetaDataLoader` — wraps DataLoader to produce support/query splits per task
- [ ] Second-order gradients — MAML needs gradients of gradients, requires extending autograd

#### Examples
- [ ] Few-shot image classification — classify new classes from 1-5 examples
- [ ] Few-shot regression — fit a new function from a handful of points

---

### Phase 17 *(Optional)* — Continual Learning
Train models sequentially on new tasks without forgetting previous ones. Intersects naturally with meta-learning — a promising direction for novel research. Requires Phases 6 (autograd) and 14b (meta-learning) first.

#### The Problem
Standard neural networks suffer from **catastrophic forgetting** — training on task B overwrites weights learned for task A. Solving this is an open research problem.

#### Algorithms
- [ ] `EWC` — Elastic Weight Consolidation, penalizes changes to weights important for previous tasks
- [ ] `PackNet` — prunes and freezes weights after each task, allocates new capacity for new tasks
- [ ] `ProgressiveNets` — adds new columns of neurons for each task, never modifies old ones
- [ ] `MAML + Continual` — use meta-learned initialization to reduce forgetting between tasks (novel intersection)

#### Supporting Infrastructure
- [ ] `TaskMemory` — stores a small replay buffer of previous task examples
- [ ] `FisherInformation` — computes which weights matter most for a task (needed for EWC)
- [ ] Continual learning benchmarks — Split-MNIST, Permuted-MNIST, Split-CIFAR

#### Examples
- [ ] Sequential digit classification — learn digits 0-4, then 5-9, measure forgetting
- [ ] Meta-learning + continual — use MAML backbone to reduce catastrophic forgetting

#### Research Angle
Combining meta-learning (Phase 12b) with continual learning is an active open problem. A novel method here with benchmark results is realistically publishable at a workshop or conference.

---

### Phase 18 *(Optional)* — CPU Performance Optimizations
Complete Phase 1-12 before starting this. These replace the internals of existing ops without changing the API.
- [ ] Cache-friendly matmul — reorder loops so memory is read sequentially (5-10x speedup)
- [ ] OpenMP parallelism — parallelize loops across CPU cores with `#pragma omp parallel for` (4-8x speedup)
- [ ] SIMD vectorization — use AVX instructions to process 8 floats at once (4-8x speedup)
- [ ] Benchmark suite — compare naive vs optimized implementations to verify correctness and measure gains

---

### Phase 19 *(Optional)* — GPU Support (CUDA)
Requires an NVIDIA GPU and CUDA Toolkit installed. Each op gets a GPU kernel alongside the CPU version.
- [ ] Add `device` field to Tensor — `CPU` or `GPU`
- [ ] `to(device)` method — move tensor between CPU and GPU memory
- [ ] CUDA matmul kernel — each GPU thread computes one output element
- [ ] CUDA kernels for element-wise ops — add, multiply, relu, sigmoid etc.
- [ ] Automatic dispatch — ops choose CPU or GPU path based on tensor device
- [ ] cuBLAS integration — use NVIDIA's optimized BLAS library for matmul (100x+ speedup)
- [ ] Update CMakeLists.txt to compile `.cu` files with `nvcc`

---

### Phase 20 *(Optional, TBD)* — Agentic Interface
To be defined once the core library is complete. Could involve an AI agent that uses the library as a tool, automatic hyperparameter search, neural architecture search, or self-tuning training loops. Revisit this phase when we get there.

---

## File Structure
```
ml_library/
├── include/ml/
│   ├── tensor.hpp              done
│   ├── ops.hpp
│   ├── loss.hpp
│   ├── dataloader.hpp
│   ├── nn/
│   │   ├── module.hpp
│   │   ├── linear.hpp
│   │   ├── activations.hpp
│   │   ├── dropout.hpp
│   │   ├── batchnorm.hpp
│   │   ├── layernorm.hpp
│   │   ├── sequential.hpp
│   │   ├── conv2d.hpp
│   │   ├── pooling.hpp
│   │   ├── flatten.hpp
│   │   ├── rnn.hpp
│   │   ├── lstm.hpp
│   │   ├── gru.hpp
│   │   ├── attention.hpp
│   │   ├── transformer.hpp
│   │   └── embedding.hpp
│   └── optim/
│       ├── optimizer.hpp
│       ├── sgd.hpp
│       ├── adam.hpp
│       └── rmsprop.hpp
├── src/
│   ├── tensor.cpp
│   ├── ops.cpp
│   ├── loss.cpp
│   ├── dataloader.cpp
│   ├── nn/
│   │   ├── linear.cpp
│   │   ├── activations.cpp
│   │   ├── dropout.cpp
│   │   ├── batchnorm.cpp
│   │   └── sequential.cpp
│   └── optim/
│       ├── sgd.cpp
│       ├── adam.cpp
│       └── rmsprop.cpp
├── python/
│   └── bindings.cpp
├── tests/
│   ├── test_tensor.cpp
│   ├── test_ops.cpp
│   ├── test_layers.cpp
│   └── test_optimizers.cpp
├── metrics/
│   ├── classification.hpp
│   └── regression.hpp
└── examples/
    ├── xor_example.py
    ├── mnist_linear.py
    ├── mnist_conv.py
    ├── regression.py
    ├── binary_classifier.py
    ├── sentiment_rnn.py
    └── text_generation.py
```

---