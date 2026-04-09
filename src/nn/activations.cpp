#include "ml/nn/activations.hpp"
#include "ml/ops.hpp"

Tensor ReLU::forward(Tensor& input) {
    return relu(input);
}

std::vector<Tensor*> ReLU::parameters() {
    return {};
}

Tensor Sigmoid::forward(Tensor& input) {
    return sigmoid(input);
}

std::vector<Tensor*> Sigmoid::parameters() {
    return {};
}

Tensor Tanh::forward(Tensor& input) {
    return tanh_op(input);
}

std::vector<Tensor*> Tanh::parameters(){
    return {};
}

Tensor Softmax::forward(Tensor& input){
    return softmax(input);
}

std::vector<Tensor*> Softmax::parameters(){
    return {};
}
