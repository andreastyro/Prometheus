#include "ml/tensor.hpp"
#include "ml/nn/linear.hpp"
#include "ml/nn/sequential.hpp"
#include "ml/nn/activations.hpp"
#include "ml/utils/model_io.hpp"
#include <stdio.h>

using namespace std;

int main(){

    // build a small model
    Sequential model({
        new Linear(2, 4),
        new ReLU(),
        new Linear(4, 1)
    });

    auto params = model.parameters();

    printf("=== Original weights ===\n");
    for (int i = 0; i < params.size(); i++){
        printf("Param %d shape: [", i);
        for (int j = 0; j < params[i]->shape.size(); j++){
            printf("%d", params[i]->shape[j]);
            if (j < params[i]->shape.size() - 1) printf(", ");
        }
        printf("] first value: %.4f\n", params[i]->data[0]);
    }

    // save
    save("tests/assets/model.bin", params);
    printf("\nSaved to tests/assets/model.bin\n");

    // load
    auto loaded = load("tests/assets/model.bin");
    printf("\n=== Loaded weights ===\n");
    for (int i = 0; i < loaded.size(); i++){
        printf("Param %d shape: [", i);
        for (int j = 0; j < loaded[i]->shape.size(); j++){
            printf("%d", loaded[i]->shape[j]);
            if (j < loaded[i]->shape.size() - 1) printf(", ");
        }
        printf("] first value: %.4f\n", loaded[i]->data[0]);
    }

    // verify
    printf("\n=== Verification ===\n");
    bool all_match = true;
    for (int i = 0; i < params.size(); i++){
        for (int j = 0; j < params[i]->num_el(); j++){
            if (params[i]->data[j] != loaded[i]->data[j]){
                printf("MISMATCH at param %d element %d\n", i, j);
                all_match = false;
            }
        }
    }
    if (all_match) printf("All weights match!\n");

    return 0;
}
