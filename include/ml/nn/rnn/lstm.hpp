#pragma once
#include "ml/nn/module.hpp"

/// Long Short-Term Memory layer.
///
/// Solves the vanishing gradient problem of plain RNNs by adding a cell state (c)
/// that carries information across many timesteps without degrading.
/// Four learned gates control what to remember, forget, and output each step:
///
///   i (input gate)  — how much new information to write into the cell
///   f (forget gate) — how much of the old cell state to keep
///   o (output gate) — how much of the cell to expose as the hidden state
///   g (cell gate)   — the candidate new information (tanh, range -1 to 1)
///
///   c_t = f * c_{t-1}  +  i * g    (additive update — no vanishing gradient)
///   h_t = o * tanh(c_t)
///
/// Input:  [seq_len, batch, input_size]
/// Output: [seq_len, batch, hidden_size]
class LSTM : public Module {
public:
    TensorPtr W_xi, W_xf, W_xo, W_xg; ///< Input weights [input_size, hidden_size] — one per gate
    TensorPtr W_hi, W_hf, W_ho, W_hg; ///< Hidden weights [hidden_size, hidden_size] — one per gate
    TensorPtr b_i, b_f, b_o, b_g;     ///< Biases [hidden_size] — one per gate

    int input_size;  ///< Size of each input vector per timestep
    int hidden_size; ///< Size of the hidden state and cell state vectors

    /// @param input_size   size of each input vector per timestep
    /// @param hidden_size  size of the hidden and cell state vectors
    LSTM(int input_size, int hidden_size);

    /// Full forward — returns output, final hidden state, and final cell state.
    /// @param input  [seq_len, batch, input_size]
    /// @param h0     initial hidden state [batch, hidden_size], or nullptr for zeros
    /// @param c0     initial cell state [batch, hidden_size], or nullptr for zeros
    /// @return {output [seq_len, batch, hidden_size], h_n, c_n}
    std::tuple<TensorPtr, TensorPtr, TensorPtr> forward(TensorPtr input, TensorPtr h0, TensorPtr c0);

    /// Module interface — h0 and c0 default to zeros, returns output sequence only.
    TensorPtr forward(TensorPtr input) override;

    std::vector<TensorPtr> parameters() override; ///< Returns all 12 tensors (4 input, 4 hidden, 4 bias)
};
