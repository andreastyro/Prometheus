#include "ml/nn/linear.hpp"
#include "ml/ops.hpp"

using namespace std;

Linear::Linear(int in_features, int out_features)
    : weights(Tensor::randn({in_features, out_features})),
      bias(Tensor::zeros({out_features})) {
}

Tensor Linear::forward(Tensor& input){
    Tensor result = matmul(input, weights);
    return broadcast_add(result, bias);
}

vector<Tensor*> Linear::parameters(){
    return {&weights, &bias};
}