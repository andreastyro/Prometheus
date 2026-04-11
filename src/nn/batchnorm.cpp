#include "ml/nn/batchnorm.hpp"
#include "ml/ops.hpp"
#include <cmath>
#include <string>

using namespace std;

BatchNorm::BatchNorm(int num_features_, float eps_, bool training_){
    
    gamma = Tensor::ones({num_features_});
    beta = Tensor::zeros({num_features_});
    eps = eps_;
    training = training_;
    
}

Tensor BatchNorm::forward(Tensor& input){
    Tensor result(input.shape);

    return;
}