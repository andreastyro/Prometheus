#pragma once
#include "ml/nn/module.hpp"

class RNN : public Module {
public:
    TensorPtr W_x;   // [input_size, hidden_size]
    TensorPtr W_h;   // [hidden_size, hidden_size]
    TensorPtr bias;  // [hidden_size]
    int input_size, hidden_size;

    RNN(int input_size, int hidden_size);

    // Full forward — returns {output [seq_len, batch, hidden_size], h_n [batch, hidden_size]}
    std::pair<TensorPtr, TensorPtr> forward(TensorPtr input, TensorPtr h0);

    // Module interface — returns output only, h0=zeros
    TensorPtr forward(TensorPtr input) override;

    std::vector<TensorPtr> parameters() override;
};
