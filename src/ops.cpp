#include "../include/ml/tensor.hpp"

using namespace std;

// Element-wise addition: a + b
Tensor add(Tensor& a, Tensor& b) {
    if (a.shape == b.shape){

        Tensor sum(a.shape);

        for(int i = 0; i < a.num_el(); i++){
            sum.data[i] = a.data[i] + b.data[i];
        }

        return sum;

    } else {
        throw runtime_error("Tensors must have the same shape for element-wise addition.");
    }
}

// Element-wise multiplication: a * b
Tensor multiply(Tensor& a, Tensor& b) {

    if (a.shape == b.shape){

        Tensor product(a.shape);

        for(int i = 0; i < a.num_el(); i++){
            product.data[i] = a.data[i] * b.data[i];
        }

        return product;

    } else{
        throw runtime_error("Tensors must have the same shape for element-wise multiplication.");
    }
}

// Matrix multiplication: a x b
Tensor matmul(Tensor& a, Tensor& b) {

    if (a.shape[1] == b.shape[0]){

        Tensor mat_prod({a.shape[0], b.shape[1]});

        for(int i = 0; i < a.shape[0]; i++){
            for(int j = 0; j < b.shape[1]; j++){
                for(int k = 0; k < a.shape[1]; k++){
                    mat_prod.data[i * b.shape[1] + j] += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
                }
            }
        }

        return mat_prod;

    } else if (b.shape[1] == a.shape[0]) {
        // user passed them in the wrong order, swap and retry
        return matmul(b, a);

    } else {
        throw runtime_error("matmul: a.cols must equal b.rows, got " + to_string(a.shape[1]) + " and " + to_string(b.shape[0]));
    }

}

// ReLU activation: max(0, x) for each element
Tensor relu(Tensor& a) {

}

// Sigmoid activation: 1 / (1 + e^-x) for each element
Tensor sigmoid(Tensor& a) {

}
