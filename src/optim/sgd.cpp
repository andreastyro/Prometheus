#include "ml/tensor.hpp"
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

void SGD::step(){

    for (int i = 0; i < parameters.size(); i++){
        auto &p = parameters[i];
        for(int j = 0; j < p->num_el(); j++){

            if (momentum == 0.0f){
                p->data[j] -= lr * p->grad[j];
                
            } else {
                velocity[i][j] = momentum * velocity[i][j] + p->grad[j];
                p->data[j] -= lr * velocity[i][j];
            }

        }
    }

}
