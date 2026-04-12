#include "ml/nn/linear.hpp"
#include "ml/ops.hpp"

using namespace std;

Linear::Linear(int in_features, int out_features) {
    weights = Tensor::randn({in_features, out_features});
    bias = Tensor::zeros({out_features});
    weights->requires_grad = true;
    bias->requires_grad = true;
}

TensorPtr Linear::forward(TensorPtr input){
    auto result = matmul(input, weights);
    return broadcast_add(result, bias);
}

vector<TensorPtr> Linear::parameters(){
    return {weights, bias};
}
