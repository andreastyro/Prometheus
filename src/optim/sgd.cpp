#include "ml/tensor.hpp"
#include "ml/autograd.hpp"
#include "ml/optim/sgd.hpp"
#include <stdio.h>
#include <algorithm>
#include <vector>

using namespace std;

SGD::SGD(vector<TensorPtr> params, float lr){
    parameters = params;
    this->lr = lr;
}

void SGD::step(){
    for (auto& p : parameters){
        for (int i = 0; i < p->num_el(); i++){
            p->data[i] -= lr * p->grad[i];
        }
    }
}
