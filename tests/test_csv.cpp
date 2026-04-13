#include "ml/data/csv.hpp"
#include <stdio.h>

using namespace std;

int main(){

    auto [x, y] = read_csv("tests/test.csv", -1, true);

    printf("x shape: [%d, %d]\n", x->shape[0], x->shape[1]);
    printf("y shape: [%d, %d]\n", y->shape[0], y->shape[1]);

    printf("\nx data:\n");
    for (int i = 0; i < x->shape[0]; i++){
        for (int j = 0; j < x->shape[1]; j++)
            printf("%.1f ", x->data[i * x->shape[1] + j]);
        printf("\n");
    }

    printf("\ny data:\n");
    for (int i = 0; i < y->shape[0]; i++)
        printf("%.1f\n", y->data[i]);

    return 0;
}
