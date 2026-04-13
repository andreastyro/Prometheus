#include "ml/nn/linear.hpp"
#include "ml/ops.hpp"
#include <cmath>

using namespace std;

Linear::Linear(int in_features, int out_features, string init) {
    weights = Tensor::randn({in_features, out_features});

    if (init == "xavier"){
        float std = sqrt(2.0f / (in_features + out_features));
        for (auto& v : weights->data) v *= std;
    } else if (init == "kaiming"){
        float std = sqrt(2.0f / in_features);
        for (auto& v : weights->data) v *= std;
    }

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
