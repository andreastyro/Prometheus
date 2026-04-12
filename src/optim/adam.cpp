#include "ml/tensor.hpp"
#include "ml/autograd.hpp"
#include "ml/optim/adam.hpp"
#include <stdio.h>
#include <algorithm>
#include <vector>

using namespace std;

Adam::Adam(vector<Tensor*> parameters, float lr, float beta1, float beta2, float eps){
    this->parameters = parameters;
    this->lr = lr;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->eps = eps;
    this->t = 0;
    
    for (Tensor* p : parameters){
        m.push_back(vector<float>(p->num_el(), 0.0f));
        v.push_back(vector<float>(p->num_el(), 0.0f));
    }

}

void Adam::step(){

    float m_hat, v_hat;

    t += 1;

    for (int i = 0; i < parameters.size(); i++){

        Tensor* p = parameters[i];

        for (int j = 0; j < p->num_el(); j++){
            m[i][j] = beta1 * m[i][j] + (1 - beta1) * p->grad[j];
            v[i][j] = beta2 * v[i][j] + (1 - beta2) * (p->grad[j] * p->grad[j]);

            m_hat = m[i][j] / (1 - pow(beta1, t));
            v_hat = v[i][j] / (1 - pow(beta2, t));

            p->data[j] -= lr * m_hat / (sqrt(v_hat) + eps);

        }

    }
}