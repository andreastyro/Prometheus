#pragma once
#include "ml/nn/module.hpp"

/// Gated Recurrent Unit layer.
///
/// A streamlined alternative to LSTM with one fewer gate and no separate cell state.
/// Merges the forget and input gates into a single update gate.
/// Matches LSTM performance on most tasks while training faster.
///
///   r (reset gate)  — how much of the old hidden state to use for the new candidate
///   z (update gate) — blend ratio between old hidden state and new candidate
///   n (new gate)    — the candidate new hidden state
///
///   h_t = (1 - z) * n  +  z * h_{t-1}    (interpolate old and new)
///
/// Input:  [seq_len, batch, input_size]
/// Output: [seq_len, batch, hidden_size]
class GRU : public Module {
public:
    TensorPtr W_xr, W_xz, W_xn; ///< Input weights [input_size, hidden_size] — one per gate
    TensorPtr W_hr, W_hz, W_hn; ///< Hidden weights [hidden_size, hidden_size] — one per gate
    TensorPtr b_r, b_z, b_n;   ///< Biases [hidden_size] — one per gate

    int input_size;  ///< Size of each input vector per timestep
    int hidden_size; ///< Size of the hidden state vector

    /// @param input_size   size of each input vector per timestep
    /// @param hidden_size  size of the hidden state vector
    GRU(int input_size, int hidden_size);

    /// Full forward — returns output sequence and final hidden state.
    /// @param input  [seq_len, batch, input_size]
    /// @param h0     initial hidden state [batch, hidden_size], or nullptr for zeros
    /// @return {output [seq_len, batch, hidden_size], h_n [batch, hidden_size]}
    std::pair<TensorPtr, TensorPtr> forward(TensorPtr input, TensorPtr h0);

    /// Module interface — h0 defaults to zeros, returns output sequence only.
    TensorPtr forward(TensorPtr input) override;

    std::vector<TensorPtr> parameters() override; ///< Returns all 9 tensors (3 input, 3 hidden, 3 bias)
};
