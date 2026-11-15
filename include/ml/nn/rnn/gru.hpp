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
/// Two forms of the candidate. The default applies the reset gate to h before
/// the hidden product:
///   n = tanh(x W_xn + (r * h) W_hn + b_n)
/// reset_after = true is PyTorch's (and cuDNN's, Keras's default) form, with
/// an extra hidden bias b_hn:
///   n = tanh(x W_xn + b_n + r * (h W_hn + b_hn))
/// It takes one hidden product a step instead of two -- faster, most of all
/// on the GPU -- and loads PyTorch nn.GRU weights exactly (b_r = b_ir + b_hr,
/// b_z = b_iz + b_hz, b_n = b_in, b_hn = b_hn; weights transposed).
///
/// Input:  [seq_len, batch, input_size]
/// Output: [seq_len, batch, hidden_size]
class GRU : public Module {
public:
    TensorPtr W_xr, W_xz, W_xn; ///< Input weights [input_size, hidden_size] — one per gate
    TensorPtr W_hr, W_hz, W_hn; ///< Hidden weights [hidden_size, hidden_size] — one per gate
    TensorPtr b_r, b_z, b_n;   ///< Biases [hidden_size] — one per gate
    TensorPtr b_hn;            ///< reset_after only: the hidden-side bias inside r * (h W_hn + b_hn)

    int input_size;  ///< Size of each input vector per timestep
    int hidden_size; ///< Size of the hidden state vector
    bool reset_after; ///< PyTorch's form of the candidate (see above)

    /// @param input_size   size of each input vector per timestep
    /// @param hidden_size  size of the hidden state vector
    /// @param reset_after  PyTorch's form: the reset gate after the hidden product
    GRU(int input_size, int hidden_size, bool reset_after = false);

    /// Full forward — returns output sequence and final hidden state.
    /// @param input  [seq_len, batch, input_size]
    /// @param h0     initial hidden state [batch, hidden_size], or nullptr for zeros
    /// @return {output [seq_len, batch, hidden_size], h_n [batch, hidden_size]}
    std::pair<TensorPtr, TensorPtr> forward(TensorPtr input, TensorPtr h0);

    /// Module interface — h0 defaults to zeros, returns output sequence only.
    TensorPtr forward(TensorPtr input) override;

    /// The 9 tensors (3 input, 3 hidden, 3 bias), then b_hn with reset_after.
    std::vector<TensorPtr> parameters() override;
};
