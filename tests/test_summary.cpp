#include "ml/nn/linear.hpp"
#include "ml/nn/activations.hpp"
#include "ml/nn/dropout.hpp"
#include "ml/nn/sequential.hpp"
#include "ml/utils/summary.hpp"
#include "ml/utils/testing.hpp"
#include <stdio.h>

using namespace std;

int main(){
    printf("=== Sequential model ===\n");
    // Linear(2,4): 2*4 + 4  = 12
    // Linear(4,8): 4*8 + 8  = 40
    // Linear(8,1): 8*1 + 1  =  9
    //                  total = 61
    Sequential model({
        new Linear(2, 4),
        new ReLU(),
        new Linear(4, 8),
        new Sigmoid(),
        new Dropout(0.5f),
        new Linear(8, 1)
    });
    model_summary(model.layers);

    int total = 0;
    for (auto& p : model.parameters()) total += p->num_el();
    CHECK_NEAR("layer count",      (double)model.layers.size(), 6, 0.0);
    CHECK_NEAR("parameter tensors", (double)model.parameters().size(), 6, 0.0);
    CHECK_NEAR("total parameters",  total, 61, 0.0);

    printf("\n=== Manual model ===\n");
    // Linear(4,8): 4*8 + 8 = 40 ; Linear(8,2): 8*2 + 2 = 18 ; total 58
    Linear l1(4, 8);
    ReLU relu;
    Linear l2(8, 2);
    Softmax softmax;
    model_summary({&l1, &relu, &l2, &softmax});

    int manual = 0;
    for (auto& p : l1.parameters()) manual += p->num_el();
    for (auto& p : l2.parameters()) manual += p->num_el();
    CHECK_NEAR("manual model parameters", manual, 58, 0.0);
    CHECK_NEAR("activations have no parameters",
               (double)(relu.parameters().size() + softmax.parameters().size()), 0, 0.0);

    return test_report();
}
