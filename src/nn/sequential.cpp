#include "ml/nn/linear.hpp"
#include "ml/ops.hpp"
#include "ml/nn/sequential.hpp"

using namespace std;

Sequential::Sequential(vector<Module*> layers){
    this->layers = layers;
}

TensorPtr Sequential::forward(TensorPtr input){
    TensorPtr out = layers[0]->forward(input);

    for (size_t i = 1; i < layers.size(); i++){
        out = layers[i]->forward(out);
    }

    return out;
}

vector<TensorPtr> Sequential::parameters() {
    vector<TensorPtr> params;
    for (size_t i = 0; i < layers.size(); i++){
        vector<TensorPtr> layer_params = layers[i]->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    return params;
}
