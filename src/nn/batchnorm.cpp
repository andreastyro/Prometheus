#include "ml/nn/batchnorm.hpp"
#include "ml/ops.hpp"
#include <cmath>
#include <string>

using namespace std;

BatchNorm::BatchNorm(int num_features, float eps, bool training){
    
    gamma = Tensor::ones({num_features});
    beta = Tensor::zeros({num_features});
    this->eps = eps;
    this->training = training;
    
}

Tensor BatchNorm::forward(Tensor& input){
    Tensor result(input.shape);

    return result;
}