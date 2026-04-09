#include "ml/tensor.hpp"
#include <cmath>
#include <string>

using namespace std;

// Element-wise addition: a + b
Tensor add(Tensor& a, Tensor& b) {
    if (a.shape == b.shape){

        Tensor result(a.shape);

        for(int i = 0; i < a.num_el(); i++){
            result.data[i] = a.data[i] + b.data[i];
        }

        return result;

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

    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        result.data[i] = a.data[i] > 0 ? a.data[i] : 0.0f;
    }

    return result;
}

// Sigmoid activation: 1 / (1 + e^-x) for each element
Tensor sigmoid(Tensor& a) {

    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        result.data[i] = 1.0f / (1.0f + exp(-a.data[i]));
    }

    return result;
}

// Element-wise subtraction: a - b
Tensor subtract(Tensor& a, Tensor& b) {
    
    if (a.shape == b.shape){

        Tensor difference(a.shape);

        for (int i = 0; i < a.num_el(); i++){
            difference.data[i] = a.data[i] - b.data[i];
        }

        return difference;
    } else{
        throw runtime_error("Tensors must have the same shape for element-wise subtraction.");
    }

}

// Element-wise division: a / b
Tensor divide(Tensor& a, Tensor& b) {
    if (a.shape == b.shape){

        Tensor quotient(a.shape);

        for (int i = 0; i < a.num_el(); i++){

            if (b.data[i] == 0){
                int row = i / a.shape[1];
                int col = i % a.shape[1];
                throw runtime_error("Division by zero at row " + to_string(row) + ", col " + to_string(col));
            }

            quotient.data[i] = a.data[i] / b.data[i];
        }

        return quotient;

    } else{
        throw runtime_error("Tensors must have the same shape for element-wise division.");
    }
}

// Tanh activation: (e^x - e^-x) / (e^x + e^-x)
Tensor tanh_op(Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        result.data[i] = (exp(a.data[i]) - exp(-a.data[i])) / (exp(a.data[i]) + exp(-a.data[i]));
    }

    return result;
}

// Softmax: converts logits to probabilities along each row
Tensor softmax(Tensor& a) {

    Tensor result(a.shape);

    for (int i = 0; i < a.shape[0]; i++){

        float sum = 0;

        for (int j = 0; j < a.shape[1]; j++){
            result.data[i * a.shape[1] + j] = exp(a.data[i * a.shape[1] + j]);
            sum += result.data[i * a.shape[1] + j];
        }

        for (int j = 0; j < a.shape[1]; j++){
            result.data[i * a.shape[1] + j] = result.data[i * a.shape[1] + j] / sum;
        }
    }

    return result;
}

// Element-wise natural log
Tensor log_op(Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        if (a.data[i] <= 0)
            throw runtime_error("Log of non-positive value at index " + to_string(i));
        result.data[i] = log(a.data[i]);
    }

    return result;
}

// Element-wise e^x
Tensor exp_op(Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        result.data[i] = exp(a.data[i]);
    }

    return result;
}

// Element-wise power: x^p
Tensor pow_op(Tensor& a, float p) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        result.data[i] = pow(a.data[i], p);
    }

    return result;
}

// Element-wise square root
Tensor sqrt_op(Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        if (a.data[i] < 0)
            throw runtime_error("Cannot compute square root of negative value " +
                                to_string(a.data[i]) + " at index " + to_string(i));
        result.data[i] = sqrt(a.data[i]);
    }

    return result;
}

// Element-wise absolute value
Tensor abs_op(Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        result.data[i] = abs(a.data[i]);
    }

    return result;
}

// Sum all elements (axis=-1) or along a given axis
Tensor sum(Tensor& a, int axis) {

    // global sum — returns a single value
    if (axis == -1) {
        Tensor result({1});
        for (int i = 0; i < a.num_el(); i++)
            result.data[0] += a.data[i];
        return result;

    // sum down rows — one sum per column e.g. {2,3} -> {3}
    } else if (axis == 0) {
        Tensor result({a.shape[1]});
        for (int i = 0; i < a.shape[0]; i++)
            for (int j = 0; j < a.shape[1]; j++)
                result.data[j] += a.data[i * a.shape[1] + j];
        return result;

    // sum across columns — one sum per row e.g. {2,3} -> {2}
    } else if (axis == 1) {
        Tensor result({a.shape[0]});
        for (int i = 0; i < a.shape[0]; i++)
            for (int j = 0; j < a.shape[1]; j++)
                result.data[i] += a.data[i * a.shape[1] + j];
        return result;

    } else {
        throw runtime_error("Invalid axis " + to_string(axis) + ", must be -1, 0, or 1");
    }
}

// Mean of all elements (axis=-1) or along a given axis
Tensor mean(Tensor& a, int axis) {

    Tensor s = sum(a, axis);

    // divide by number of elements that were summed
    float count;
    if      (axis == -1) count = a.num_el();
    else if (axis == 0)  count = a.shape[0];  // summed over rows
    else if (axis == 1)  count = a.shape[1];  // summed over cols
    else throw runtime_error("Invalid axis " + to_string(axis) + ", must be -1, 0, or 1");

    for (int i = 0; i < s.num_el(); i++)
        s.data[i] /= count;

    return s;
}

// Max element — returns a scalar tensor
Tensor max_op(Tensor& a) {
    Tensor result({1});
    result.data[0] = a.data[0];
    for (int i = 1; i < a.num_el(); i++)
        if (a.data[i] > result.data[0])
            result.data[0] = a.data[i];
    return result;
}

// Min element — returns a scalar tensor
Tensor min_op(Tensor& a) {
    Tensor result({1});
    result.data[0] = a.data[0];
    for (int i = 1; i < a.num_el(); i++)
        if (a.data[i] < result.data[0])
            result.data[0] = a.data[i];
    return result;
}

// Clip — clamp all values between min and max
Tensor clip(Tensor& a, float min_val, float max_val) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        if (a.data[i] > max_val){
            result.data[i] = max_val;
        } else if (a.data[i] < min_val) {
            result.data[i] = min_val;
        } else {
            result.data[i] = a.data[i];
        }
    }

    return result;
}

// Broadcast add — adds a 1D bias tensor to every row of a 2D tensor
// e.g. a={32,64} + b={64} -> {32,64}
Tensor broadcast_add(Tensor& a, Tensor& b) {


    if (a.shape.size() != 2)
        throw runtime_error("broadcast_add: 'a' must be a 2D tensor, got " + to_string(a.shape.size()) + "D");

    if (b.shape.size() != 1)
        throw runtime_error("broadcast_add: 'b' must be a 1D tensor, got " + to_string(b.shape.size()) + "D");

    if (b.shape[0] != a.shape[1])
        throw runtime_error("broadcast_add: 'b' size " + to_string(b.shape[0]) +
                            " must match a's columns " + to_string(a.shape[1]));

    Tensor result(a.shape);

    for (int i = 0; i < a.shape[0]; i++){
        for (int j = 0; j < a.shape[1]; j++){
            result.data[i * a.shape[1] + j] = a.data[i * a.shape[1] + j] + b.data[j];
        }
    }

    return result;
    
}


