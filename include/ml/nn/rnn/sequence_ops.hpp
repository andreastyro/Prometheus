#pragma once
/// Differentiable timestep slicing and stacking for the recurrent layers.
///
/// RNN, LSTM and GRU all walk a sequence one timestep at a time. Both ends of
/// that walk used to be done with raw buffer copies:
///
///     x_t->data()[...]   = input->data()[...];    // slice  input[t]
///     output->data()[...] = h->data()[...];       // stack  h_t into output[t]
///
/// A raw copy produces a tensor with no grad_fn, so the gradient could not flow
/// from the output sequence back into the per-timestep graph, nor from the
/// timestep inputs back into the original input tensor. Every parameter
/// gradient came back as exactly zero when the layer was used through the
/// Module interface. These two helpers do the same work while wiring the result
/// into the graph.

#include "ml/tensor.hpp"
#include <vector>

/// input: [seq_len, batch, features]. Returns input[t] as [batch, features],
/// with a backward that scatters the slice gradient back into input.
TensorPtr rnn_slice_step(TensorPtr input, int t);

/// steps: per-timestep states, each [batch, hidden]. Returns the stacked
/// sequence [seq_len, batch, hidden], with a backward that routes each
/// timestep slice of the output gradient back into the matching state.
TensorPtr rnn_stack_steps(const std::vector<TensorPtr>& steps);
