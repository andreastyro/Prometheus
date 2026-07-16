#pragma once
#include "ml/nn/module.hpp"

/// Basic Recurrent Neural Network layer.
///
/// Processes a sequence one timestep at a time. At each step it takes the
/// current input and its previous hidden state, and produces a new hidden state:
///   h_t = tanh( x_t @ W_x  +  h_{t-1} @ W_h  +  bias )
///
/// The hidden state carries a summary of everything seen so far in the sequence.
/// Limitation: tends to forget early timesteps in long sequences — use LSTM or GRU instead.
///
/// Input:  [seq_len, batch, input_size]
/// Output: [seq_len, batch, hidden_size]
class RNN : public Module {
public:
    TensorPtr W_x;  ///< [input_size, hidden_size] — weights for the current input
    TensorPtr W_h;  ///< [hidden_size, hidden_size] — weights for the previous hidden state
    TensorPtr bias; ///< [hidden_size]

    int input_size;  ///< Size of each input vector at a single timestep
    int hidden_size; ///< Size of the hidden state vector

    /// @param input_size   size of each input vector per timestep
    /// @param hidden_size  size of the hidden state vector
    RNN(int input_size, int hidden_size);

    /// Full forward — returns output sequence and final hidden state.
    /// @param input  [seq_len, batch, input_size]
    /// @param h0     initial hidden state [batch, hidden_size], or nullptr for zeros
    /// @return {output [seq_len, batch, hidden_size], h_n [batch, hidden_size]}
    std::pair<TensorPtr, TensorPtr> forward(TensorPtr input, TensorPtr h0);

    /// Module interface — runs with h0=zeros, returns output sequence only.
    TensorPtr forward(TensorPtr input) override;

    std::vector<TensorPtr> parameters() override; ///< Returns {W_x, W_h, bias}
};
