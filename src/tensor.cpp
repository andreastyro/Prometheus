#include "../include/ml/tensor.hpp"
#include <stdio.h>

void Tensor::print() const {
    int cols = shape[1];

    for (int i = 0; i < (int)data.size(); i++) {
        printf("%.2f ", data[i]);

        // start a new line after every row
        if ((i + 1) % cols == 0)
            printf("\n");
    }
}
