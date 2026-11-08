#include "ml/nn/dropout.hpp"
#include "ml/ops.hpp"
#include "ml/tensor.hpp"
#include <memory>
#include <vector>

using namespace std;

Dropout::Dropout(float rate, bool training){
    this->rate = rate;
    this->training = training;
}

// Inverted dropout: during training each element is zeroed with probability
// `rate` and the survivors are scaled by 1/(1-rate), so the expected value is
// unchanged and inference needs no rescaling.
//
// The backward pass must apply the same mask: an element that was dropped
// contributes nothing to the loss and so receives no gradient. This previously
// returned a fresh tensor with no grad_fn, which severed the graph and left
// every layer upstream of the Dropout with a gradient of exactly zero.
TensorPtr Dropout::forward(TensorPtr input){
    if (training == false)
        return input;

    uniform_real_distribution<float> dist(0.0f, 1.0f);

    const float scale = 1.0f / (1.0f - rate);
    const int   n     = input->num_el();

    auto result = make_shared<Tensor>(input->shape());

    // mask[i] is the factor actually applied, so the backward pass can reuse it
    vector<float> mask(n);

    for (int i = 0; i < n; i++) {
        const bool drop = dist(prometheus_rng()) < rate;
        mask[i] = drop ? 0.0f : scale;
        result->data()[i] = input->data()[i] * mask[i];
    }

    if (input->requires_grad()) {
        auto node = make_node(result, {input});
        node->name = "Dropout::forward";
        autograd::attach_reference(*node, [mtensor = make_shared<Tensor>(input->shape(), mask)](const std::vector<TensorPtr>& in) -> TensorPtr {
            return multiply(in[0], mtensor);
        });

        node->backward_fn = [input, mask, n](const TensorPtr& result) {
            for (int i = 0; i < n; i++)
                input->grad()[i] += result->grad()[i] * mask[i];
        };
    }

    return result;
}

vector<TensorPtr> Dropout::parameters(){
    return {};
}
