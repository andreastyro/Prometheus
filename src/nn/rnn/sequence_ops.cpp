#include "ml/nn/rnn/sequence_ops.hpp"
#include "ml/ops.hpp"
#include <stdexcept>

using namespace std;

TensorPtr rnn_slice_step(TensorPtr input, int t) {
    if (input->shape().size() != 3)
        throw runtime_error("rnn_slice_step: expected a 3D input "
                            "[seq_len, batch, features]");

    const int seq_len  = input->shape()[0];
    const int batch    = input->shape()[1];
    const int features = input->shape()[2];

    if (t < 0 || t >= seq_len)
        throw runtime_error("rnn_slice_step: timestep out of range");

    auto out = make_shared<Tensor>(Shape{batch, features});

    const int base = t * batch * features;
    for (int i = 0; i < batch * features; i++)
        out->data()[i] = input->data()[base + i];

    if (input->requires_grad()) {
        auto node = make_node(out, {input});
        node->name = "rnn_slice_step";
        autograd::attach_reference(*node, [base, batch, features](const std::vector<TensorPtr>& in) -> TensorPtr {
            auto idx = make_shared<vector<int64_t>>((size_t)batch * features);
            for (int64_t i = 0; i < (int64_t)batch * features; i++) (*idx)[i] = base + i;
            return gather_flat(in[0], idx, {(int64_t)batch, (int64_t)features});
        });
        node->backward_fn = [input, base, batch, features](const TensorPtr& out) {
            for (int i = 0; i < batch * features; i++)
                input->grad()[base + i] += out->grad()[i];
        };
    }

    return out;
}

TensorPtr rnn_stack_steps(const vector<TensorPtr>& steps) {
    if (steps.empty())
        throw runtime_error("rnn_stack_steps: no timesteps to stack");

    const int seq_len = (int)steps.size();
    const int batch   = steps[0]->shape()[0];
    const int hidden  = steps[0]->shape()[1];
    const int stride  = batch * hidden;

    auto out = make_shared<Tensor>(Shape{seq_len, batch, hidden});

    bool any_grad = false;
    for (int t = 0; t < seq_len; t++) {
        if (steps[t]->num_el() != stride)
            throw runtime_error("rnn_stack_steps: timestep shapes differ");
        for (int i = 0; i < stride; i++)
            out->data()[t * stride + i] = steps[t]->data()[i];
        if (steps[t]->requires_grad()) any_grad = true;
    }

    if (any_grad) {
        auto node = make_node(out, steps);
        node->name = "rnn_stack_steps";
        autograd::attach_reference(*node, [seq_len, batch, hidden](const std::vector<TensorPtr>& in) -> TensorPtr {
            vector<TensorPtr> flat;
            for (auto& t : in) flat.push_back(reshape(t, {t->numel()}));
            return reshape(concat_op(flat, 0), {(int64_t)seq_len, (int64_t)batch, (int64_t)hidden});
        });
        node->backward_fn = [steps, seq_len, stride](const TensorPtr& out) {
            for (int t = 0; t < seq_len; t++) {
                if (!steps[t]->requires_grad()) continue;
                for (int i = 0; i < stride; i++)
                    steps[t]->grad()[i] += out->grad()[t * stride + i];
            }
        };
    }

    return out;
}
