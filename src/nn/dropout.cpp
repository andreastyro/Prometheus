#include "ml/nn/dropout.hpp"
#include <random>

using namespace std;

Dropout::Dropout(float rate, bool training){
    this->rate = rate;
    this->training = training;
}

Tensor Dropout::forward(Tensor& input){
    if (training == false)
        return input;

    mt19937 rng(random_device{}());
    uniform_real_distribution<float> dist(0.0f, 1.0f);

    float scale = 1.0f / (1.0f - rate);

    Tensor result(input.shape);

    for (int i = 0; i < input.num_el(); i++) {
        if (dist(rng) < rate)
            result.data[i] = 0.0f;
        else
            result.data[i] = input.data[i] * scale;
    }
    return result;
}

std::vector<Tensor*> Dropout::parameters(){
    return {};
}
