#pragma once
#include <vector>
#include "ml/tensor.hpp"

/// One kernel for a whole recurrent layer: RNN, LSTM or GRU over every
/// timestep, as a single graph node with a hand-written backward through time.
///
/// The layers used to build the recurrence out of generic ops -- per timestep,
/// a separate small matmul for every gate and a dozen elementwise ops, each
/// allocating and each recorded as its own graph node: an LSTM over 64 steps
/// made about 1,500 nodes. Here, as in cuDNN and PyTorch:
///
///   - the input projection of every timestep is one gemm,
///     [seq*batch, in] @ [in, gates*H], with the gate weights packed side by side
///   - each step is one gemm for the recurrence plus a fused elementwise kernel
///   - backward runs the recurrence in reverse once, then the weight and input
///     gradients over all timesteps are again one gemm each
///
/// The formulas are the ones these layers have always used, which for the GRU
/// is not PyTorch's: the reset gate applies before the hidden matmul,
///   n = tanh(x W_xn + (r * h) W_hn + b_n).
namespace rnn_kernel {

enum class Cell { RNN, LSTM, GRU };

struct Result {
    TensorPtr output;   ///< [seq, batch, H]
    TensorPtr h_n;      ///< [batch, H] -- the last step of output
    TensorPtr c_n;      ///< [batch, H] for an LSTM, null otherwise
};

/// input [seq, batch, in]; h0 / c0 [batch, H] or null for zeros.
/// Wx[k] is [in, H], Wh[k] is [H, H] and b[k] is [H], one per gate in the
/// cell's order: RNN {h}, LSTM {i, f, o, g}, GRU {r, z, n}.
Result run(Cell cell, TensorPtr input, TensorPtr h0, TensorPtr c0, int H,
           const std::vector<TensorPtr>& Wx, const std::vector<TensorPtr>& Wh,
           const std::vector<TensorPtr>& b);

} // namespace rnn_kernel
