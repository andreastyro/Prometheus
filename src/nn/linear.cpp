#include "ml/nn/linear.hpp"
#include "ml/ops.hpp"
#include <cmath>
#include <random>

using namespace std;

// Every value drawn from U(-bound, bound): PyTorch's default for Linear and
// the convolutions (kaiming_uniform with a = sqrt(5) works out to
// bound = 1/sqrt(fan_in), and the bias uses the same bound).
static void fill_uniform(const TensorPtr& t, float bound) {
    std::uniform_real_distribution<float> u(-bound, bound);
    for (auto& v : t->values()) v = u(prometheus_rng());
}

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
    if (init == "default") {
        // as nn.Linear. This used to leave the weights at N(0, 1), which no
        // network of any depth trains from: activations grow by sqrt(in) a layer
        const float bound = 1.0f / sqrt((float)in_features);
        fill_uniform(weights, bound);
        fill_uniform(bias, bound);
    }
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
