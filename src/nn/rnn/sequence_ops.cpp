#include "ml/nn/rnn/sequence_ops.hpp"
#include <stdexcept>

using namespace std;

TensorPtr rnn_slice_step(TensorPtr input, int t) {
    if (input->shape.size() != 3)
        throw runtime_error("rnn_slice_step: expected a 3D input "
                            "[seq_len, batch, features]");

    const int seq_len  = input->shape[0];
    const int batch    = input->shape[1];
    const int features = input->shape[2];

    if (t < 0 || t >= seq_len)
        throw runtime_error("rnn_slice_step: timestep out of range");

    auto out = make_shared<Tensor>(vector<int>{batch, features});

    const int base = t * batch * features;
    for (int i = 0; i < batch * features; i++)
        out->data[i] = input->data[base + i];

    if (input->requires_grad) {
        auto node = make_node(out, {input});
        node->backward_fn = [input, out, base, batch, features]() {
            for (int i = 0; i < batch * features; i++)
                input->grad[base + i] += out->grad[i];
        };
    }

    return out;
}

TensorPtr rnn_stack_steps(const vector<TensorPtr>& steps) {
    if (steps.empty())
        throw runtime_error("rnn_stack_steps: no timesteps to stack");

    const int seq_len = (int)steps.size();
    const int batch   = steps[0]->shape[0];
    const int hidden  = steps[0]->shape[1];
    const int stride  = batch * hidden;

    auto out = make_shared<Tensor>(vector<int>{seq_len, batch, hidden});

    bool any_grad = false;
    for (int t = 0; t < seq_len; t++) {
        if (steps[t]->num_el() != stride)
            throw runtime_error("rnn_stack_steps: timestep shapes differ");
        for (int i = 0; i < stride; i++)
            out->data[t * stride + i] = steps[t]->data[i];
        if (steps[t]->requires_grad) any_grad = true;
    }

    if (any_grad) {
        auto node = make_node(out, steps);
        node->backward_fn = [steps, out, seq_len, stride]() {
            for (int t = 0; t < seq_len; t++) {
                if (!steps[t]->requires_grad) continue;
                for (int i = 0; i < stride; i++)
                    steps[t]->grad[i] += out->grad[t * stride + i];
            }
        };
    }

    return out;
}
