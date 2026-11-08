#include "ml/nn/linear.hpp"
#include "ml/ops.hpp"
#include <cmath>

using namespace std;

Linear::Linear(int in_features, int out_features, string init) {
    weights = Tensor::randn({in_features, out_features});

    if (init == "xavier"){
        float std = sqrt(2.0f / (in_features + out_features));
        for (auto& v : weights->values()) v *= std;
    } else if (init == "kaiming"){
        float std = sqrt(2.0f / in_features);
        for (auto& v : weights->values()) v *= std;
    }

    bias = Tensor::zeros({out_features});
    weights->set_requires_grad(true);
    bias->set_requires_grad(true);
}

TensorPtr Linear::forward(TensorPtr input){
    // one op: the bias is folded into the gemm's output instead of a
    // separate broadcast_add pass over the result
    return linear_op(input, weights, bias);
}

vector<TensorPtr> Linear::parameters(){
    return {weights, bias};
}
