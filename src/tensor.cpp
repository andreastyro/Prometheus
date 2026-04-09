#include "../include/ml/tensor.hpp"
#include <stdio.h>
#include <random>
#include <stdexcept>
#include <string>

using namespace std;

void Tensor::print() const {
    int cols = shape[1];

    for (int i = 0; i < (int)data.size(); i++) {
        printf("%.2f ", data[i]);

        // start a new line after every row
        if ((i + 1) % cols == 0)
            printf("\n");
    }
}

// Random normal values (mean=0, std=1)
Tensor Tensor::randn(vector<int> shape) {
    Tensor t(shape);
    random_device rd;                        // seed from hardware
    mt19937 gen(rd());                       // mersenne twister generator
    normal_distribution<float> dist(0.0f, 1.0f); // mean=0, std=1
    for (float& x : t.data)
        x = dist(gen);
    return t;
}

// Flip rows and cols (2D only)
Tensor Tensor::transpose() const {

    if (shape.size() != 2)
        throw runtime_error("transpose: only supported for 2D tensors");
    
    int rows = shape[0], cols = shape[1];
    Tensor result({cols, rows});
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[j * rows + i] = data[i * cols + j];
    return result;
}

// Change shape without changing data
Tensor Tensor::reshape(vector<int> new_shape) const {
    int new_total = 1;
    for (int i : new_shape)
        new_total *= i;

    if (new_total != num_el())
        throw runtime_error("reshape: cannot reshape tensor of size " + to_string(num_el()) +
                            " into shape with size " + to_string(new_total));

    Tensor result(new_shape);
    result.data = data;
    return result;
}
