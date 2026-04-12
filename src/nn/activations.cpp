#include "ml/nn/activations.hpp"
#include "ml/ops.hpp"

using namespace std;

TensorPtr ReLU::forward(TensorPtr input) {
    return relu(input);
}

vector<TensorPtr> ReLU::parameters() {
    return {};
}

TensorPtr Sigmoid::forward(TensorPtr input) {
    return sigmoid(input);
}

vector<TensorPtr> Sigmoid::parameters() {
    return {};
}

TensorPtr Tanh::forward(TensorPtr input) {
    return tanh_op(input);
}

vector<TensorPtr> Tanh::parameters(){
    return {};
}

TensorPtr Softmax::forward(TensorPtr input){
    return softmax(input);
}

vector<TensorPtr> Softmax::parameters(){
    return {};
}
