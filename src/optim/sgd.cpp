#include "ml/tensor.hpp"
#include <cmath>
#include "ml/autograd.hpp"
#include "ml/optim/sgd.hpp"
#include <stdio.h>
#include <algorithm>
#include <vector>

using namespace std;

SGD::SGD(vector<TensorPtr> params, float lr, float momentum){
    parameters = params;
    this->lr = lr;
    this->momentum = momentum;

    if (momentum != 0.0f){
        for (auto p : parameters){
            velocity.push_back(vector<float>(p->num_el(), 0.0f));
        }
    } 

}

// Parallel over elements, with each buffer's pointer and size hoisted: the
// loop used to call num_el() in its condition and index the velocity through
// a vector of vectors on every element.
void SGD::step(){
    for (int i = 0; i < (int)parameters.size(); i++){
        auto& p = parameters[i];
        const int n = (int)p->numel();
        float* w = p->data();
        const float* g = p->grad();
        const float lr_ = lr, mu = momentum;
        if (mu == 0.0f) {
            #pragma omp parallel for schedule(static) if(n >= 32768)
            for (int j = 0; j < n; j++) w[j] -= lr_ * g[j];
        } else {
            float* v = velocity[i].data();
            #pragma omp parallel for schedule(static) if(n >= 32768)
            for (int j = 0; j < n; j++) {
                v[j] = mu * v[j] + g[j];
                w[j] -= lr_ * v[j];
            }
        }
    }
}
