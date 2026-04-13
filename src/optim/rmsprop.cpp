#include "ml/tensor.hpp"
#include "ml/autograd.hpp"
#include "ml/optim/rmsprop.hpp"
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

void RMSprop::step(){
    for (int i = 0; i < parameters.size(); i++){
        
        auto &p = parameters[i];

        for (int j = 0; j < p->num_el(); j++){
            
            v[i][j] = beta * v[i][j] + (1 - beta) * (pow(p->grad[j], 2));
            p->data[j] -= lr * p->grad[j] / (sqrt(v[i][j]) + eps);   
        }
    }
}