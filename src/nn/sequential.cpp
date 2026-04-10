#include "ml/nn/linear.hpp"
#include "ml/ops.hpp"
#include "ml/nn/sequential.hpp"

using namespace std;

Sequential::Sequential(vector<Module*> layers_){
    layers = layers_;
}

Tensor Sequential::forward(Tensor &input){

    Tensor out = layers[0]->forward(input);

    for (int i = 0; i < layers.size(); i++){
        out = layers[i]->forward(out);
    }

    return out;
}

vector<Tensor*> Sequential::parameters() {
    vector<Tensor*> params;
    for (int i = 0; i < layers.size(); i++){

        vector<Tensor*> layer_params = layers[i]->parameters();

        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    return params;
}