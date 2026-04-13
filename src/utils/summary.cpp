#include "ml/utils/summary.hpp"
#include <stdio.h>

using namespace std;

void model_summary(vector<Module*> layers){
    printf("%-20s %-20s %s\n", "Layer", "Output Shape", "Params");
    printf("------------------------------------------------------------\n");

    int total_params = 0;

    for (int i = 0; i < layers.size(); i++){
        auto params = layers[i]->parameters();

        int layer_params = 0;
        for (auto& p : params)
            layer_params += p->num_el();

        // get layer type name and output shape
        string name = "Layer " + to_string(i);
        string shape_str = "-";

        if (auto l = dynamic_cast<Linear*>(layers[i])){
            name = "Linear";
            int in_feat = l->parameters()[0]->shape[0];
            int out_feat = l->parameters()[0]->shape[1];
            shape_str = "[" + to_string(in_feat) + " -> " + to_string(out_feat) + "]";
        } else if (dynamic_cast<ReLU*>(layers[i]))
            name = "ReLU";
        else if (dynamic_cast<Sigmoid*>(layers[i]))
            name = "Sigmoid";
        else if (dynamic_cast<Tanh*>(layers[i]))
            name = "Tanh";
        else if (dynamic_cast<Softmax*>(layers[i]))
            name = "Softmax";
        else if (dynamic_cast<Dropout*>(layers[i]))
            name = "Dropout";

        printf("%-20s %-20s %d\n", name.c_str(), shape_str.c_str(), layer_params);
        total_params += layer_params;
    }

    printf("------------------------------------------------------------\n");
    printf("Total params: %d\n", total_params);
}
