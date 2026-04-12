#include "ml/tensor.hpp"
#include "ml/autograd.hpp"
#include <cmath>
#include <string>

using namespace std;

// Helper: set up a GradNode and attach it to result
shared_ptr<GradNode> make_node(Tensor& result, vector<Tensor*> inputs) {
    result.requires_grad = true;
    auto node = make_shared<GradNode>();
    node->inputs = inputs;
    result.grad_fn = node;
    return node;
}

// Element-wise addition: a + b
Tensor add(Tensor& a, Tensor& b) {
    if (a.shape == b.shape){

        Tensor result(a.shape);

        for(int i = 0; i < a.num_el(); i++){
            result.data[i] = a.data[i] + b.data[i];
        }

        if (a.requires_grad || b.requires_grad) {
            auto node = make_node(result, {&a, &b});

            // forward logic ends here

            node->backward_fn = [&a, &b, &result](){ // set backward function for add
                for (int i = 0; i < result.num_el(); i++){
                    if (a.requires_grad) a.grad[i] += result.grad[i]; // Add grad of previous operation's result
                    if (b.requires_grad) b.grad[i] += result.grad[i];
                }
            };

        }
        
        return result;

    } else {
        throw runtime_error("Tensors must have the same shape for element-wise addition.");
    }
}

// Scalar + tensor
Tensor add(float scalar, Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++)
        result.data[i] = scalar + a.data[i];

    if (a.requires_grad) {
        auto node = make_node(result, {&a});
        node->backward_fn = [&a, &result]() {
            for (int i = 0; i < result.num_el(); i++)
                a.grad[i] += result.grad[i];
        };
    }

    return result;
}

// Element-wise multiplication: a * b
Tensor multiply(Tensor& a, Tensor& b) {

    if (a.shape == b.shape){

        Tensor result(a.shape);

        for(int i = 0; i < a.num_el(); i++){
            result.data[i] = a.data[i] * b.data[i];
        }

        if (a.requires_grad || b.requires_grad) {
            auto node = make_node(result, {&a, &b});
            node->backward_fn = [&a, &b, &result](){
                for (int i = 0; i < result.num_el(); i++){
                    if (a.requires_grad) a.grad[i] += b.data[i] * result.grad[i];
                    if (b.requires_grad) b.grad[i] += a.data[i] * result.grad[i];
                }
            };
        }

        return result;

    } else{
        throw runtime_error("Tensors must have the same shape for element-wise multiplication.");
    }
}

// Scalar * tensor
Tensor multiply(float scalar, Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++)
        result.data[i] = scalar * a.data[i];

    if (a.requires_grad) {
        auto node = make_node(result, {&a});
        node->backward_fn = [&a, &result, scalar]() {
            for (int i = 0; i < result.num_el(); i++)
                a.grad[i] += scalar * result.grad[i];
        };
    }

    return result;
}

// Matrix multiplication: a x b
Tensor matmul(Tensor& a, Tensor& b) {

    if (a.shape[1] == b.shape[0]){

        Tensor result({a.shape[0], b.shape[1]});

        for(int i = 0; i < a.shape[0]; i++){
            for(int j = 0; j < b.shape[1]; j++){
                for(int k = 0; k < a.shape[1]; k++){
                    result.data[i * b.shape[1] + j] += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
                }
            }
        }

        if (a.requires_grad || b.requires_grad){
            auto node = make_node(result, {&a, &b});

            node->backward_fn = [&a, &b, &result](){

                Tensor grad_out({a.shape[0], b.shape[1]}, result.grad);           

                if (a.requires_grad){
                    Tensor bT = b.transpose();
                    Tensor dA = matmul(grad_out, bT);
                    for (int i = 0; i < a.num_el(); i++) a.grad[i] += dA.data[i];
                }

                if (b.requires_grad){
                    Tensor aT = a.transpose();
                    Tensor dB = matmul(aT, grad_out);
                    for (int i = 0; i < b.num_el(); i++) b.grad[i] += dB.data[i];
                }

            };

        }

        return result;

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

    if (a.requires_grad) {
        auto node = make_node(result, {&a});
        node->backward_fn = [&a, &result]() {
            for (int i = 0; i < result.num_el(); i++)
                if (a.data[i] > 0) a.grad[i] += result.grad[i];
        };
    }

    return result;
}

// Sigmoid activation: 1 / (1 + e^-x) for each element
Tensor sigmoid(Tensor& a) {

    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        result.data[i] = 1.0f / (1.0f + exp(-a.data[i]));
    }

    if (a.requires_grad){
        auto node = make_node(result, {&a});

        node->backward_fn = [&a, &result](){

            for (int i = 0; i < result.num_el(); i++){
                a.grad[i] += result.data[i] * (1 - result.data[i]) * result.grad[i];
            }

        };

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

        if (a.requires_grad || b.requires_grad) {
            auto node = make_node(difference, {&a, &b});
            node->backward_fn = [&a, &b, &difference]() {
                for (int i = 0; i < difference.num_el(); i++) {
                    if (a.requires_grad) a.grad[i] += difference.grad[i];
                    if (b.requires_grad) b.grad[i] -= difference.grad[i];
                }
            };
        }

        return difference;
    } else{
        throw runtime_error("Tensors must have the same shape for element-wise subtraction.");
    }

}

// Scalar - tensor  (d/da = -1)
Tensor subtract(float scalar, Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++)
        result.data[i] = scalar - a.data[i];

    if (a.requires_grad) {
        auto node = make_node(result, {&a});
        node->backward_fn = [&a, &result]() {
            for (int i = 0; i < result.num_el(); i++)
                a.grad[i] -= result.grad[i];
        };
    }

    return result;
}

// Tensor - scalar  (d/da = 1)
Tensor subtract(Tensor& a, float scalar) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++)
        result.data[i] = a.data[i] - scalar;

    if (a.requires_grad) {
        auto node = make_node(result, {&a});
        node->backward_fn = [&a, &result]() {
            for (int i = 0; i < result.num_el(); i++)
                a.grad[i] += result.grad[i];
        };
    }

    return result;
}

// Element-wise division: a / b
Tensor divide(Tensor& a, Tensor& b) {
    if (a.shape == b.shape){

        Tensor result(a.shape);

        for (int i = 0; i < a.num_el(); i++){

            if (b.data[i] == 0){
                int row = i / a.shape[1];
                int col = i % a.shape[1];
                throw runtime_error("Division by zero at row " + to_string(row) + ", col " + to_string(col));
            }

            result.data[i] = a.data[i] / b.data[i];
        }

        if (a.requires_grad || b.requires_grad){

            auto node = make_node(result, {&a, &b});

            node->backward_fn = [&a, &b, &result](){

                for (int i = 0; i < result.num_el(); i++){
                    if (a.requires_grad) a.grad[i] += result.grad[i] / b.data[i];
                    if (b.requires_grad) b.grad[i] += -result.grad[i] * a.data[i] / (b.data[i] * b.data[i]); 
                }

            };

        }

        return result;

    } else{
        throw runtime_error("Tensors must have the same shape for element-wise division.");
    }
}

// Scalar / tensor  dL/da = -scalar * result.grad / a^2
Tensor divide(float scalar, Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++) {
        if (a.data[i] == 0)
            throw std::runtime_error("Division by zero at index " + std::to_string(i));
        result.data[i] = scalar / a.data[i];
    }

    if (a.requires_grad) {
        auto node = make_node(result, {&a});
        node->backward_fn = [&a, &result, scalar]() {
            for (int i = 0; i < result.num_el(); i++)
                a.grad[i] += -scalar * result.grad[i] / (a.data[i] * a.data[i]);
        };
    }

    return result;
}

// Tensor / scalar  dL/da = result.grad / scalar
Tensor divide(Tensor& a, float scalar) {
    if (scalar == 0)
        throw std::runtime_error("Division by zero");
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++)
        result.data[i] = a.data[i] / scalar;

    if (a.requires_grad) {
        auto node = make_node(result, {&a});
        node->backward_fn = [&a, &result, scalar]() {
            for (int i = 0; i < result.num_el(); i++)
                a.grad[i] += result.grad[i] / scalar;
        };
    }

    return result;
}

// Tanh activation: (e^x - e^-x) / (e^x + e^-x)
Tensor tanh_op(Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        result.data[i] = (exp(a.data[i]) - exp(-a.data[i])) / (exp(a.data[i]) + exp(-a.data[i]));
    }

    if (a.requires_grad){
        auto node = make_node(result, {&a});

        node->backward_fn = [&a, &result](){
            for (int i = 0; i < result.num_el(); i++){
                a.grad[i] += (1 - result.data[i] * result.data[i]) * result.grad[i];
            }
        };
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

    if (a.requires_grad){

        auto node = make_node(result, {&a});
        int rows = a.shape[0], cols = a.shape[1];

        node->backward_fn = [&a, &result, cols, rows](){

            Tensor jacobian({cols, cols});

            for (int i = 0; i < rows; i++){
                
                // fill jacobian for row i

                for (int j = 0; j < cols; j++){
                    for (int k = 0; k < cols; k++){
                        float s_j = result.data[i * cols + j];
                        float s_k = result.data[i * cols + k];

                        jacobian.data[j * cols + k] = s_j * ((j == k ? 1.0f : 0.0f) - s_k);
                    }
                }

                for (int j = 0; j < cols; j++){
                    for (int k = 0; k < cols; k++){
                        a.grad[i * cols + j] += jacobian.data[j * cols + k] * result.grad[i * cols + k];
                    }
                }

            }
        };

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

    if (a.requires_grad) {
        auto node = make_node(result, {&a});
        node->backward_fn = [&a, &result]() {
            for (int i = 0; i < result.num_el(); i++)
                a.grad[i] += result.grad[i] / a.data[i];  // d(ln x)/dx = 1/x
        };
    }

    return result;
}

// Element-wise e^x
Tensor exp_op(Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        result.data[i] = exp(a.data[i]);
    }

    if (a.requires_grad) {
        auto node = make_node(result, {&a});
        node->backward_fn = [&a, &result]() {
            for (int i = 0; i < result.num_el(); i++)
                a.grad[i] += result.data[i] * result.grad[i];  // d(e^x)/dx = e^x = result
        };
    }

    return result;
}

// Element-wise power: x^p
Tensor pow_op(Tensor& a, float p) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        result.data[i] = pow(a.data[i], p);
    }

    if (a.requires_grad) {
        auto node = make_node(result, {&a});
        node->backward_fn = [&a, &result, p]() {
            for (int i = 0; i < result.num_el(); i++)
                a.grad[i] += p * pow(a.data[i], p - 1) * result.grad[i];  // d(x^p)/dx = p*x^(p-1)
        };
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

    if (a.requires_grad) {
        auto node = make_node(result, {&a});
        node->backward_fn = [&a, &result]() {
            for (int i = 0; i < result.num_el(); i++)
                a.grad[i] += result.grad[i] / (2.0f * result.data[i]);  // d(sqrt(x))/dx = 1/(2*sqrt(x))
        };
    }

    return result;
}

// Element-wise absolute value
Tensor abs_op(Tensor& a) {
    Tensor result(a.shape);
    for (int i = 0; i < a.num_el(); i++){
        result.data[i] = abs(a.data[i]);
    }

    if (a.requires_grad){
        auto node = make_node(result, {&a});

        node->backward_fn = [&a, &result](){
            for (int i = 0; i < result.num_el(); i++){
                if (a.data[i] > 0) a.grad[i] += result.grad[i];
                else if (a.data[i] < 0) a.grad[i] -= result.grad[i];
            }
        };

    }

    return result;
}

// Sum all elements (axis=-1) or along a given axis
Tensor sum(Tensor& a, int axis) {

    int rows = a.shape[0], cols = a.shape[1];

    // global sum — returns a single value
    if (axis == -1) {
        Tensor result({1});
        for (int i = 0; i < a.num_el(); i++)
            result.data[0] += a.data[i];

        if (a.requires_grad) {
            auto node = make_node(result, {&a});
            node->backward_fn = [&a, &result]() {
                for (int i = 0; i < a.num_el(); i++)
                    a.grad[i] += result.grad[0];  // every element gets the same gradient
            };
        }
        return result;

    // sum down rows — one sum per column e.g. {2,3} -> {3}
    } else if (axis == 0) {
        Tensor result({cols});
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[j] += a.data[i * cols + j];

        if (a.requires_grad) {
            auto node = make_node(result, {&a});
            node->backward_fn = [&a, &result, rows, cols]() {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a.grad[i * cols + j] += result.grad[j];  // each element gets its column's gradient
            };
        }
        return result;

    // sum across columns — one sum per row e.g. {2,3} -> {2}
    } else if (axis == 1) {
        Tensor result({rows});
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[i] += a.data[i * cols + j];

        if (a.requires_grad) {
            auto node = make_node(result, {&a});
            node->backward_fn = [&a, &result, rows, cols]() {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a.grad[i * cols + j] += result.grad[i];  // each element gets its row's gradient
            };
        }
        return result;

    } else {
        throw runtime_error("Invalid axis " + to_string(axis) + ", must be -1, 0, or 1");
    }
}

// Mean of all elements (axis=-1) or along a given axis
Tensor mean(Tensor& a, int axis) {

    int rows = a.shape[0], cols = a.shape[1];

    float count;
    if      (axis == -1) count = a.num_el();
    else if (axis == 0)  count = rows;
    else if (axis == 1)  count = cols;
    else throw runtime_error("Invalid axis " + to_string(axis) + ", must be -1, 0, or 1");

    Tensor result = sum(a, axis);
    for (int i = 0; i < result.num_el(); i++)
        result.data[i] /= count;

    if (a.requires_grad) {

        auto node = make_node(result, {&a});

        node->backward_fn = [&a, &result, axis, count, rows, cols]() {
            if (axis == -1) {
                for (int i = 0; i < a.num_el(); i++)
                    a.grad[i] += result.grad[0] / count;

            } else if (axis == 0) {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a.grad[i * cols + j] += result.grad[j] / count; // mean of column

            } else if (axis == 1) {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a.grad[i * cols + j] += result.grad[i] / count; // mean of row
            }
        };
    }

    return result;
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

    if (a.requires_grad || b.requires_grad) {
        auto node = make_node(result, {&a, &b});
        node->backward_fn = [&a, &b, &result]() {
            int rows = a.shape[0], cols = a.shape[1];

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (a.requires_grad)
                        a.grad[i * cols + j] += result.grad[i * cols + j];

                    if (b.requires_grad)
                        b.grad[j] += result.grad[i * cols + j];  // accumulate over all rows
                }
            }
        };
    }

    return result;

}


