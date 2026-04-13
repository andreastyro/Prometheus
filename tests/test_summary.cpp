#include "ml/nn/linear.hpp"
#include "ml/nn/activations.hpp"
#include "ml/nn/dropout.hpp"
#include "ml/nn/sequential.hpp"
#include "ml/utils/summary.hpp"
#include <stdio.h>

using namespace std;

int main(){

    printf("=== Sequential model ===\n");
    Sequential model({
        new Linear(2, 4),
        new ReLU(),
        new Linear(4, 8),
        new Sigmoid(),
        new Dropout(0.5f),
        new Linear(8, 1)
    });
    model_summary(model.layers);

    printf("\n=== Manual model ===\n");
    Linear l1(4, 8);
    ReLU relu;
    Linear l2(8, 2);
    Softmax softmax;
    model_summary({&l1, &relu, &l2, &softmax});

    return 0;
}
