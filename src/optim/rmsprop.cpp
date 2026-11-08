#include "ml/tensor.hpp"
#include <cmath>
#include "ml/autograd.hpp"
#include "ml/optim/rmsprop.hpp"
#include "../ops_internal.hpp"
#include <stdio.h>
#include <algorithm>
#include <vector>

using namespace std;

RMSprop::RMSprop(vector<TensorPtr> params, float lr, float beta, float eps){
    parameters = params;
    this->lr = lr;
    this->beta = beta;
    this->eps = eps;

    for (auto &p : parameters){
        v.push_back(vector<float>(p->num_el(), 0.0f));
    }

}

// g*g rather than pow(g, 2): pow is a general double-precision power call,
// and running it on every gradient made this step 22x slower than PyTorch's.
void RMSprop::step(){
    const float b = beta, lr_ = lr, eps_ = eps;
    for (int i = 0; i < (int)parameters.size(); i++){
        auto& p = parameters[i];
        if (p->device() == Device::CUDA) {   // the update runs on the GPU
            const float* gd = device_grad(p);
            if (!gd) continue;
            cuda_ops::rmsprop_step(device_weights(p), gd, v_dev.get(i, v[i]), p->numel(), lr_, b, eps_);
            continue;
        }
        const int n = (int)p->numel();
        float* w = p->data();
        const float* g = p->grad();
        float* V = v[i].data();
        #pragma omp parallel for schedule(static) if(n >= 32768)
        for (int j = 0; j < n; j++){
            const float gj = g[j];
            V[j] = b * V[j] + (1.0f - b) * gj * gj;
            w[j] -= lr_ * gj / (std::sqrt(V[j]) + eps_);
        }
    }
}
