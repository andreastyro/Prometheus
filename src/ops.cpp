#include "ml/tensor.hpp"
#include "ml/autograd.hpp"
#include "ml/ops.hpp"
#include "ml/matmul_backend.hpp"
#include "ml/fastmath.hpp"
#include "ml/strided.hpp"
#include <cstring>
#include <cmath>
#include <string>

using namespace std;

// Every op below reads its inputs through data(), the raw row-major pointer.
// checked_input() makes that safe: a non-contiguous view is materialised
// (the copy joins the graph, so gradients still reach the view's source), and
// a dtype or device the kernels cannot read raises instead of being
// reinterpreted as floats.
TensorPtr checked_input(const char* op, TensorPtr t) {
    if (!t) return t;
    if (t->device() != Device::CPU)
        throw runtime_error(string(op) + ": tensor is on " + device_name(t->device()) +
                            "; this build computes on cpu only");
    if (t->dtype() != DType::Float32)
        throw runtime_error(string(op) + ": expects float32, got " + dtype_name(t->dtype()) +
                            "; convert with .to(float32)");
    return t->is_contiguous() ? t : contiguous(t);
}



// Element-wise addition: a + b
TensorPtr add(TensorPtr a, TensorPtr b) {
    a = checked_input("add", a);
    b = checked_input("add", b);
    if (a->shape() == b->shape()){

        auto result = make_shared<Tensor>(a->shape());

        const int n_1 = a->num_el();
        #pragma omp parallel for schedule(static) if(n_1 >= 32768)
        for (int i = 0; i < n_1; i++){
            result->data()[i] = a->data()[i] + b->data()[i];
        }

        if (a->requires_grad() || b->requires_grad()) {
            auto node = make_node(result, {a, b});

            node->backward_fn = [a, b, result](){
                const int n_2 = result->num_el();
                #pragma omp parallel for schedule(static) if(n_2 >= 32768)
                for (int i = 0; i < n_2; i++){
                    if (a->requires_grad()) a->grad()[i] += result->grad()[i];
                    if (b->requires_grad()) b->grad()[i] += result->grad()[i];
                }
            };
        }

        return result;

    } else {
        throw runtime_error("Tensors must have the same shape for element-wise addition.");
    }
}

// Scalar + tensor
TensorPtr add(float scalar, TensorPtr a) {
    a = checked_input("add", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_3 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_3 >= 32768)
    for (int i = 0; i < n_3; i++)
        result->data()[i] = scalar + a->data()[i];

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            const int n_4 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_4 >= 32768)
            for (int i = 0; i < n_4; i++)
                a->grad()[i] += result->grad()[i];
        };
    }

    return result;
}

// Element-wise multiplication: a * b
TensorPtr multiply(TensorPtr a, TensorPtr b) {
    a = checked_input("multiply", a);
    b = checked_input("multiply", b);

    if (a->shape() == b->shape()){

        auto result = make_shared<Tensor>(a->shape());

        const int n_5 = a->num_el();
        #pragma omp parallel for schedule(static) if(n_5 >= 32768)
        for (int i = 0; i < n_5; i++){
            result->data()[i] = a->data()[i] * b->data()[i];
        }

        if (a->requires_grad() || b->requires_grad()) {
            auto node = make_node(result, {a, b});
            node->backward_fn = [a, b, result](){
                const int n_6 = result->num_el();
                #pragma omp parallel for schedule(static) if(n_6 >= 32768)
                for (int i = 0; i < n_6; i++){
                    if (a->requires_grad()) a->grad()[i] += b->data()[i] * result->grad()[i];
                    if (b->requires_grad()) b->grad()[i] += a->data()[i] * result->grad()[i];
                }
            };
        }

        return result;

    } else{
        throw runtime_error("Tensors must have the same shape for element-wise multiplication.");
    }
}

// Scalar * tensor
TensorPtr multiply(float scalar, TensorPtr a) {
    a = checked_input("multiply", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_7 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_7 >= 32768)
    for (int i = 0; i < n_7; i++)
        result->data()[i] = scalar * a->data()[i];

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, scalar]() {
            const int n_8 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_8 >= 32768)
            for (int i = 0; i < n_8; i++)
                a->grad()[i] += scalar * result->grad()[i];
        };
    }

    return result;
}

// Matrix multiplication: a x b
TensorPtr matmul(TensorPtr a, TensorPtr b) {
    a = checked_input("matmul", a);
    b = checked_input("matmul", b);

    if (a->shape()[1] == b->shape()[0]){

        auto result = make_shared<Tensor>(Shape{a->shape()[0], b->shape()[1]});

        matmul_forward(a->shape()[0], b->shape()[1], a->shape()[1],
                       a->data(), b->data(), result->data());

        if (a->requires_grad() || b->requires_grad()){
            auto node = make_node(result, {a, b});

            node->backward_fn = [a, b, result](){
                int a_rows = a->shape()[0], inner = a->shape()[1], b_cols = b->shape()[1];

                if (a->requires_grad())
                    matmul_backward_a(a_rows, inner, b_cols,
                                      result->grad(), b->data(), a->grad());

                if (b->requires_grad())
                    matmul_backward_b(a_rows, inner, b_cols,
                                      result->grad(), a->data(), b->grad());
            };
        }

        return result;

    } else if (b->shape()[1] == a->shape()[0]) {
        // user passed them in the wrong order, swap and retry
        return matmul(b, a);

    } else {
        throw runtime_error("matmul: a.cols must equal b.rows, got " + to_string(a->shape()[1]) + " and " + to_string(b->shape()[0]));
    }
}

// ReLU activation: max(0, x) for each element
TensorPtr relu(TensorPtr a) {
    a = checked_input("relu", a);

    auto result = make_shared<Tensor>(a->shape());
    const int n_9 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_9 >= 32768)
    for (int i = 0; i < n_9; i++){
        result->data()[i] = a->data()[i] > 0 ? a->data()[i] : 0.0f;
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            const int n_10 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_10 >= 32768)
            for (int i = 0; i < n_10; i++)
                if (a->data()[i] > 0) a->grad()[i] += result->grad()[i];
        };
    }

    return result;
}

// Sigmoid activation: 1 / (1 + e^-x) for each element
TensorPtr sigmoid(TensorPtr a) {
    a = checked_input("sigmoid", a);

    auto result = make_shared<Tensor>(a->shape());
    const int n_11 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_11 >= 32768)
    for (int i = 0; i < n_11; i++){
        result->data()[i] = 1.0f / (1.0f + exp(-a->data()[i]));
    }

    if (a->requires_grad()){
        auto node = make_node(result, {a});

        node->backward_fn = [a, result](){
            const int n_12 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_12 >= 32768)
            for (int i = 0; i < n_12; i++){
                a->grad()[i] += result->data()[i] * (1 - result->data()[i]) * result->grad()[i];
            }
        };
    }

    return result;
}

// Element-wise subtraction: a - b
TensorPtr subtract(TensorPtr a, TensorPtr b) {
    a = checked_input("subtract", a);
    b = checked_input("subtract", b);

    if (a->shape() == b->shape()){

        auto difference = make_shared<Tensor>(a->shape());

        const int n_13 = a->num_el();
        #pragma omp parallel for schedule(static) if(n_13 >= 32768)
        for (int i = 0; i < n_13; i++){
            difference->data()[i] = a->data()[i] - b->data()[i];
        }

        if (a->requires_grad() || b->requires_grad()) {
            auto node = make_node(difference, {a, b});
            node->backward_fn = [a, b, difference]() {
                const int n_14 = difference->num_el();
                #pragma omp parallel for schedule(static) if(n_14 >= 32768)
                for (int i = 0; i < n_14; i++) {
                    if (a->requires_grad()) a->grad()[i] += difference->grad()[i];
                    if (b->requires_grad()) b->grad()[i] -= difference->grad()[i];
                }
            };
        }

        return difference;
    } else{
        throw runtime_error("Tensors must have the same shape for element-wise subtraction.");
    }
}

// Scalar - tensor  (d/da = -1)
TensorPtr subtract(float scalar, TensorPtr a) {
    a = checked_input("subtract", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_15 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_15 >= 32768)
    for (int i = 0; i < n_15; i++)
        result->data()[i] = scalar - a->data()[i];

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            const int n_16 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_16 >= 32768)
            for (int i = 0; i < n_16; i++)
                a->grad()[i] -= result->grad()[i];
        };
    }

    return result;
}

// Tensor - scalar  (d/da = 1)
TensorPtr subtract(TensorPtr a, float scalar) {
    a = checked_input("subtract", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_17 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_17 >= 32768)
    for (int i = 0; i < n_17; i++)
        result->data()[i] = a->data()[i] - scalar;

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            const int n_18 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_18 >= 32768)
            for (int i = 0; i < n_18; i++)
                a->grad()[i] += result->grad()[i];
        };
    }

    return result;
}

// Element-wise division: a / b
TensorPtr divide(TensorPtr a, TensorPtr b) {
    a = checked_input("divide", a);
    b = checked_input("divide", b);
    if (a->shape() == b->shape()){

        auto result = make_shared<Tensor>(a->shape());

        const int n_19 = a->num_el();
        #pragma omp parallel for schedule(static) if(n_19 >= 32768)
        for (int i = 0; i < n_19; i++){

            if (b->data()[i] == 0){
                int row = i / a->shape()[1];
                int col = i % a->shape()[1];
                throw runtime_error("Division by zero at row " + to_string(row) + ", col " + to_string(col));
            }

            result->data()[i] = a->data()[i] / b->data()[i];
        }

        if (a->requires_grad() || b->requires_grad()){

            auto node = make_node(result, {a, b});

            node->backward_fn = [a, b, result](){
                const int n_20 = result->num_el();
                #pragma omp parallel for schedule(static) if(n_20 >= 32768)
                for (int i = 0; i < n_20; i++){
                    if (a->requires_grad()) a->grad()[i] += result->grad()[i] / b->data()[i];
                    if (b->requires_grad()) b->grad()[i] += -result->grad()[i] * a->data()[i] / (b->data()[i] * b->data()[i]);
                }
            };
        }

        return result;

    } else{
        throw runtime_error("Tensors must have the same shape for element-wise division.");
    }
}

// Scalar / tensor  dL/da = -scalar * result.grad / a^2
TensorPtr divide(float scalar, TensorPtr a) {
    a = checked_input("divide", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_21 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_21 >= 32768)
    for (int i = 0; i < n_21; i++) {
        if (a->data()[i] == 0)
            throw std::runtime_error("Division by zero at index " + std::to_string(i));
        result->data()[i] = scalar / a->data()[i];
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, scalar]() {
            const int n_22 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_22 >= 32768)
            for (int i = 0; i < n_22; i++)
                a->grad()[i] += -scalar * result->grad()[i] / (a->data()[i] * a->data()[i]);
        };
    }

    return result;
}

// Tensor / scalar  dL/da = result.grad / scalar
TensorPtr divide(TensorPtr a, float scalar) {
    a = checked_input("divide", a);
    if (scalar == 0)
        throw std::runtime_error("Division by zero");
    auto result = make_shared<Tensor>(a->shape());
    const int n_23 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_23 >= 32768)
    for (int i = 0; i < n_23; i++)
        result->data()[i] = a->data()[i] / scalar;

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, scalar]() {
            const int n_24 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_24 >= 32768)
            for (int i = 0; i < n_24; i++)
                a->grad()[i] += result->grad()[i] / scalar;
        };
    }

    return result;
}

// Tanh activation: (e^x - e^-x) / (e^x + e^-x)
TensorPtr tanh_op(TensorPtr a) {
    a = checked_input("tanh_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_25 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_25 >= 32768)
    for (int i = 0; i < n_25; i++){
        result->data()[i] = (exp(a->data()[i]) - exp(-a->data()[i])) / (exp(a->data()[i]) + exp(-a->data()[i]));
    }

    if (a->requires_grad()){
        auto node = make_node(result, {a});

        node->backward_fn = [a, result](){
            const int n_26 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_26 >= 32768)
            for (int i = 0; i < n_26; i++){
                a->grad()[i] += (1 - result->data()[i] * result->data()[i]) * result->grad()[i];
            }
        };
    }

    return result;
}

// Softmax: converts logits to probabilities along each row
TensorPtr softmax(TensorPtr a) {
    a = checked_input("softmax", a);

    auto result = make_shared<Tensor>(a->shape());

    // Subtract the row maximum before exponentiating. softmax(x + c) equals
    // softmax(x) for any constant c, so this changes nothing mathematically
    // while keeping every exponent at or below zero.
    //
    // Without it exp() overflows to inf and the result is NaN: measured here,
    // a row whose largest logit was 90 already produced NaN, and attention
    // scores or an untrained output layer reach that easily. The backward pass
    // is unaffected, since it is written in terms of the output probabilities.
    const int cols = a->shape()[1];

    for (int i = 0; i < a->shape()[0]; i++){

        float row_max = a->data()[i * cols];
        for (int j = 1; j < cols; j++)
            if (a->data()[i * cols + j] > row_max)
                row_max = a->data()[i * cols + j];

        float s = 0;

        for (int j = 0; j < cols; j++){
            result->data()[i * cols + j] = exp(a->data()[i * cols + j] - row_max);
            s += result->data()[i * cols + j];
        }

        for (int j = 0; j < cols; j++){
            result->data()[i * cols + j] = result->data()[i * cols + j] / s;
        }
    }

    if (a->requires_grad()){

        auto node = make_node(result, {a});
        int rows = a->shape()[0], cols = a->shape()[1];

        node->backward_fn = [a, result, cols, rows](){

            Tensor jacobian({cols, cols});

            for (int i = 0; i < rows; i++){

                // fill jacobian for row i
                for (int j = 0; j < cols; j++){
                    for (int k = 0; k < cols; k++){
                        float s_j = result->data()[i * cols + j];
                        float s_k = result->data()[i * cols + k];

                        jacobian.data()[j * cols + k] = s_j * ((j == k ? 1.0f : 0.0f) - s_k);
                    }
                }

                for (int j = 0; j < cols; j++){
                    for (int k = 0; k < cols; k++){
                        a->grad()[i * cols + j] += jacobian.data()[j * cols + k] * result->grad()[i * cols + k];
                    }
                }
            }
        };
    }

    return result;
}

// Element-wise natural log
TensorPtr log_op(TensorPtr a) {
    a = checked_input("log_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_27 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_27 >= 32768)
    for (int i = 0; i < n_27; i++){
        if (a->data()[i] <= 0)
            throw runtime_error("Log of non-positive value at index " + to_string(i));
        result->data()[i] = log(a->data()[i]);
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            const int n_28 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_28 >= 32768)
            for (int i = 0; i < n_28; i++)
                a->grad()[i] += result->grad()[i] / a->data()[i];  // d(ln x)/dx = 1/x
        };
    }

    return result;
}

// Element-wise e^x
TensorPtr exp_op(TensorPtr a) {
    a = checked_input("exp_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_29 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_29 >= 32768)
    for (int i = 0; i < n_29; i++){
        result->data()[i] = exp(a->data()[i]);
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            const int n_30 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_30 >= 32768)
            for (int i = 0; i < n_30; i++)
                a->grad()[i] += result->data()[i] * result->grad()[i];  // d(e^x)/dx = e^x = result
        };
    }

    return result;
}

// Element-wise power: x^p
TensorPtr pow_op(TensorPtr a, float p) {
    a = checked_input("pow_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_31 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_31 >= 32768)
    for (int i = 0; i < n_31; i++){
        result->data()[i] = pow(a->data()[i], p);
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, p]() {
            const int n_32 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_32 >= 32768)
            for (int i = 0; i < n_32; i++)
                a->grad()[i] += p * pow(a->data()[i], p - 1) * result->grad()[i];  // d(x^p)/dx = p*x^(p-1)
        };
    }

    return result;
}

// Element-wise square root
TensorPtr sqrt_op(TensorPtr a) {
    a = checked_input("sqrt_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_33 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_33 >= 32768)
    for (int i = 0; i < n_33; i++){
        if (a->data()[i] < 0)
            throw runtime_error("Cannot compute square root of negative value " +
                                to_string(a->data()[i]) + " at index " + to_string(i));
        result->data()[i] = sqrt(a->data()[i]);
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            const int n_34 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_34 >= 32768)
            for (int i = 0; i < n_34; i++)
                a->grad()[i] += result->grad()[i] / (2.0f * result->data()[i]);  // d(sqrt(x))/dx = 1/(2*sqrt(x))
        };
    }

    return result;
}

// Element-wise absolute value
TensorPtr abs_op(TensorPtr a) {
    a = checked_input("abs_op", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_35 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_35 >= 32768)
    for (int i = 0; i < n_35; i++){
        result->data()[i] = abs(a->data()[i]);
    }

    if (a->requires_grad()){
        auto node = make_node(result, {a});

        node->backward_fn = [a, result](){
            const int n_36 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_36 >= 32768)
            for (int i = 0; i < n_36; i++){
                if (a->data()[i] > 0) a->grad()[i] += result->grad()[i];
                else if (a->data()[i] < 0) a->grad()[i] -= result->grad()[i];
            }
        };
    }

    return result;
}

// Sum all elements (axis=-1) or along a given axis
TensorPtr sum(TensorPtr a, int axis) {
    a = checked_input("sum", a);

    // A 1-D tensor has no shape[1]. Reading it unconditionally was an
    // out-of-bounds vector access: harmless-looking in release builds, where it
    // returns whatever follows the shape vector, and caught immediately by a
    // debug standard library. mean(x, -1) and sum(x, -1) are legitimate on a
    // 1-D tensor and must not need a column count at all.
    const int rows = a->shape()[0];
    const int cols = (a->shape().size() > 1) ? a->shape()[1] : 1;
    if (axis != -1 && a->shape().size() < 2)
        throw runtime_error("axis " + to_string(axis) +
                            " requires a 2-D tensor; this tensor is 1-D");

    // global sum — returns a single value
    if (axis == -1) {
        auto result = make_shared<Tensor>(Shape{1});
        const int n_37 = a->num_el();
        for (int i = 0; i < n_37; i++)
            result->data()[0] += a->data()[i];

        if (a->requires_grad()) {
            auto node = make_node(result, {a});
            node->backward_fn = [a, result]() {
                const int n_38 = a->num_el();
                for (int i = 0; i < n_38; i++)
                    a->grad()[i] += result->grad()[0];  // every element gets the same gradient
            };
        }
        return result;

    // sum down rows — one sum per column e.g. {2,3} -> {3}
    } else if (axis == 0) {
        auto result = make_shared<Tensor>(Shape{cols});
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result->data()[j] += a->data()[i * cols + j];

        if (a->requires_grad()) {
            auto node = make_node(result, {a});
            node->backward_fn = [a, result, rows, cols]() {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a->grad()[i * cols + j] += result->grad()[j];  // each element gets its column's gradient
            };
        }
        return result;

    // sum across columns — one sum per row e.g. {2,3} -> {2}
    } else if (axis == 1) {
        auto result = make_shared<Tensor>(Shape{rows});
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result->data()[i] += a->data()[i * cols + j];

        if (a->requires_grad()) {
            auto node = make_node(result, {a});
            node->backward_fn = [a, result, rows, cols]() {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a->grad()[i * cols + j] += result->grad()[i];  // each element gets its row's gradient
            };
        }
        return result;

    } else {
        throw runtime_error("Invalid axis " + to_string(axis) + ", must be -1, 0, or 1");
    }
}

// Mean of all elements (axis=-1) or along a given axis
TensorPtr mean(TensorPtr a, int axis) {
    a = checked_input("mean", a);

    // A 1-D tensor has no shape[1]. Reading it unconditionally was an
    // out-of-bounds vector access: harmless-looking in release builds, where it
    // returns whatever follows the shape vector, and caught immediately by a
    // debug standard library. mean(x, -1) and sum(x, -1) are legitimate on a
    // 1-D tensor and must not need a column count at all.
    const int rows = a->shape()[0];
    const int cols = (a->shape().size() > 1) ? a->shape()[1] : 1;
    if (axis != -1 && a->shape().size() < 2)
        throw runtime_error("axis " + to_string(axis) +
                            " requires a 2-D tensor; this tensor is 1-D");

    float count;
    if      (axis == -1) count = (float)a->num_el();
    else if (axis == 0)  count = (float)rows;
    else if (axis == 1)  count = (float)cols;
    else throw runtime_error("Invalid axis " + to_string(axis) + ", must be -1, 0, or 1");

    // compute directly without calling sum to keep a single clean grad node
    shared_ptr<Tensor> result;
    if (axis == -1) {
        result = make_shared<Tensor>(Shape{1});
        const int n_39 = a->num_el();
        for (int i = 0; i < n_39; i++)
            result->data()[0] += a->data()[i];
        result->data()[0] /= count;
    } else if (axis == 0) {
        result = make_shared<Tensor>(Shape{cols});
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result->data()[j] += a->data()[i * cols + j];
        for (int j = 0; j < cols; j++)
            result->data()[j] /= count;
    } else {
        result = make_shared<Tensor>(Shape{rows});
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result->data()[i] += a->data()[i * cols + j];
        for (int i = 0; i < rows; i++)
            result->data()[i] /= count;
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});

        node->backward_fn = [a, result, axis, count, rows, cols]() {
            if (axis == -1) {
                const int n_40 = a->num_el();
                for (int i = 0; i < n_40; i++)
                    a->grad()[i] += result->grad()[0] / count;

            } else if (axis == 0) {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a->grad()[i * cols + j] += result->grad()[j] / count;

            } else if (axis == 1) {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a->grad()[i * cols + j] += result->grad()[i] / count;
            }
        };
    }

    return result;
}

// Max element — returns a scalar tensor
// Max element. The gradient reaches only the element that attained the
// maximum; every other element has no influence on the result. Ties go to the
// first occurrence, matching the forward scan.
TensorPtr max_op(TensorPtr a) {
    a = checked_input("max_op", a);
    auto result = make_shared<Tensor>(Shape{1});
    result->data()[0] = a->data()[0];
    int arg = 0;
    for (int i = 1; i < a->num_el(); i++)
        if (a->data()[i] > result->data()[0]) {
            result->data()[0] = a->data()[i];
            arg = i;
        }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, arg]() {
            a->grad()[arg] += result->grad()[0];
        };
    }

    return result;
}

// Min element — returns a scalar tensor
// Min element — mirror of max_op, gradient reaches only the arg-min.
TensorPtr min_op(TensorPtr a) {
    a = checked_input("min_op", a);
    auto result = make_shared<Tensor>(Shape{1});
    result->data()[0] = a->data()[0];
    int arg = 0;
    for (int i = 1; i < a->num_el(); i++)
        if (a->data()[i] < result->data()[0]) {
            result->data()[0] = a->data()[i];
            arg = i;
        }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, arg]() {
            a->grad()[arg] += result->grad()[0];
        };
    }

    return result;
}

// Clip — clamp all values between min and max
TensorPtr clip(TensorPtr a, float min_val, float max_val) {
    a = checked_input("clip", a);
    auto result = make_shared<Tensor>(a->shape());
    const int n_41 = a->num_el();
    #pragma omp parallel for schedule(static) if(n_41 >= 32768)
    for (int i = 0; i < n_41; i++){
        if (a->data()[i] > max_val){
            result->data()[i] = max_val;
        } else if (a->data()[i] < min_val) {
            result->data()[i] = min_val;
        } else {
            result->data()[i] = a->data()[i];
        }
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, min_val, max_val]() {
            const int n_42 = result->num_el();
            #pragma omp parallel for schedule(static) if(n_42 >= 32768)
            for (int i = 0; i < n_42; i++) {
                // gradient is 1 where value wasn't clamped, 0 where it was
                if (a->data()[i] > min_val && a->data()[i] < max_val)
                    a->grad()[i] += result->grad()[i];
            }
        };
    }

    return result;
}

// Broadcast add — adds a 1D bias tensor to every row of a 2D tensor
// e.g. a={32,64} + b={64} -> {32,64}
TensorPtr broadcast_add(TensorPtr a, TensorPtr b) {
    a = checked_input("broadcast_add", a);
    b = checked_input("broadcast_add", b);

    if (a->shape().size() != 2)
        throw runtime_error("broadcast_add: 'a' must be a 2D tensor, got " + to_string(a->shape().size()) + "D");

    if (b->shape().size() != 1)
        throw runtime_error("broadcast_add: 'b' must be a 1D tensor, got " + to_string(b->shape().size()) + "D");

    if (b->shape()[0] != a->shape()[1])
        throw runtime_error("broadcast_add: 'b' size " + to_string(b->shape()[0]) +
                            " must match a's columns " + to_string(a->shape()[1]));

    auto result = make_shared<Tensor>(a->shape());

    for (int i = 0; i < a->shape()[0]; i++){
        for (int j = 0; j < a->shape()[1]; j++){
            result->data()[i * a->shape()[1] + j] = a->data()[i * a->shape()[1] + j] + b->data()[j];
        }
    }

    if (a->requires_grad() || b->requires_grad()) {
        auto node = make_node(result, {a, b});
        node->backward_fn = [a, b, result]() {
            int rows = a->shape()[0], cols = a->shape()[1];

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (a->requires_grad())
                        a->grad()[i * cols + j] += result->grad()[i * cols + j];

                    if (b->requires_grad())
                        b->grad()[j] += result->grad()[i * cols + j];  // accumulate over all rows
                }
            }
        };
    }

    return result;
}

// Argmax — returns a 1D tensor of class indices, one per row
TensorPtr argmax(TensorPtr a) {
    a = checked_input("argmax", a);
    int rows = a->shape()[0];
    int cols = a->shape()[1];
    auto result = make_shared<Tensor>(Shape{rows});

    for (int i = 0; i < rows; i++){
        int best = 0;
        float max_val = a->data()[i * cols];
        for (int j = 1; j < cols; j++){
            if (a->data()[i * cols + j] > max_val){
                max_val = a->data()[i * cols + j];
                best = j;
            }
        }
        result->data()[i] = (float)best;
    }

    return result;
}

// GELU activation — tanh approximation used by GPT-2/3/4.
// Formula: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
TensorPtr gelu(TensorPtr a) {
    a = checked_input("gelu", a);
    auto result = make_shared<Tensor>(a->shape());
    int n = a->num_el();
    constexpr float kSqrt2OverPi = 0.7978845608f;

    // gelu was missed when the other elementwise ops were parallelised -- it
    // bounds its loop with a local, not num_el() -- and tanhf is an opaque
    // call that kept it to one lane besides
    #pragma omp parallel for schedule(static) if(n >= 32768)
    for (int i = 0; i < n; i++) {
        float x = a->data()[i];
        float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
        result->data()[i] = 0.5f * x * (1.0f + fastmath::tanh(inner));
    }

    if (a->requires_grad()) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            constexpr float k = 0.7978845608f;
            int m = a->num_el();
            #pragma omp parallel for schedule(static) if(m >= 32768)
            for (int i = 0; i < m; i++) {
                float x = a->data()[i];
                float inner = k * (x + 0.044715f * x * x * x);
                float t = fastmath::tanh(inner);
                float d_inner = k * (1.0f + 3.0f * 0.044715f * x * x);
                float grad = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * d_inner;
                a->grad()[i] += result->grad()[i] * grad;
            }
        };
    }

    return result;
}

TensorPtr concat_op(vector<TensorPtr> inputs, int axis) {
    for (auto& t_ : inputs) t_ = checked_input("concat_op", t_);
    if (inputs.empty())
        throw runtime_error("concat_op: no inputs");

    const Shape shape0 = inputs[0]->shape();
    const int ndim = (int)shape0.size();
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim)
        throw runtime_error("concat_op: axis out of range");

    // every input must match on all axes except the concatenation one
    int concat_total = 0;
    for (auto& t : inputs) {
        if ((int)t->shape().size() != ndim)
            throw runtime_error("concat_op: inputs differ in rank");
        for (int d = 0; d < ndim; d++)
            if (d != axis && t->shape()[d] != shape0[d])
                throw runtime_error("concat_op: inputs differ on axis " + to_string(d));
        concat_total += t->shape()[axis];
    }

    Shape out_shape = shape0;
    out_shape[axis] = concat_total;
    auto result = make_shared<Tensor>(out_shape);

    // Treat the tensor as [outer, axis_dim, inner]: copying a contiguous
    // inner-sized run per outer index per input is all that is needed.
    int outer = 1;
    for (int d = 0; d < axis; d++) outer *= shape0[d];
    int inner = 1;
    for (int d = axis + 1; d < ndim; d++) inner *= shape0[d];

    bool any_grad = false;
    for (auto& t : inputs) if (t->requires_grad()) any_grad = true;

    for (int o = 0; o < outer; o++) {
        int written = 0;
        for (auto& t : inputs) {
            const int span = t->shape()[axis] * inner;
            const float* src = t->data() + o * span;
            float* dst = result->data() + o * concat_total * inner + written;
            for (int i = 0; i < span; i++) dst[i] = src[i];
            written += span;
        }
    }

    if (any_grad) {
        auto node = make_node(result, inputs);
        node->backward_fn = [inputs, result, outer, inner, concat_total, axis]() {
            for (int o = 0; o < outer; o++) {
                int read = 0;
                for (auto& t : inputs) {
                    const int span = t->shape()[axis] * inner;
                    if (t->requires_grad()) {
                        const float* src = result->grad() + o * concat_total * inner + read;
                        float* dst = t->grad() + o * span;
                        for (int i = 0; i < span; i++) dst[i] += src[i];
                    }
                    read += span;
                }
            }
        };
    }

    return result;
}

TensorPtr matmul_bt(TensorPtr a, TensorPtr b) {
    a = checked_input("matmul_bt", a);
    b = checked_input("matmul_bt", b);
    if (a->shape().size() != 2 || b->shape().size() != 2)
        throw runtime_error("matmul_bt: both operands must be 2-D");

    const int m = a->shape()[0], k = a->shape()[1];
    const int n = b->shape()[0];
    if (b->shape()[1] != k)
        throw runtime_error("matmul_bt: a.cols must equal b.cols, got "
                            + to_string(k) + " and " + to_string(b->shape()[1]));

    auto result = make_shared<Tensor>(Shape{m, n});
    matmul_forward_bt(m, n, k, a->data(), b->data(),
                      result->data());

    if (a->requires_grad() || b->requires_grad()) {
        auto node = make_node(result, {a, b});

        node->backward_fn = [a, b, result, m, n, k]() {
            const float* g = result->grad();

            // C = A @ B^T  =>  dA = dC @ B,  dB = dC^T @ A
            if (a->requires_grad())
                matmul_accumulate(m, k, n, g, b->data(), a->grad());

            if (b->requires_grad())
                matmul_backward_b(m, n, k, a->data(), g, b->grad());
        };
    }

    return result;
}

// ── Views ────────────────────────────────────────────────────────────────────
//
// A view shares its source's storage and differs only in shape, strides and
// offset, so taking one is O(1) and writes through it reach the source.
// Gradients follow the convention in tensor.hpp -- every grad buffer is
// row-major over its own tensor's shape -- so a view's backward only has to
// map positions between two row-major layouts.

static int wrap_dim(int d, int nd, const char* op) {
    const int w = d < 0 ? d + nd : d;
    if (w < 0 || w >= nd)
        throw runtime_error(string(op) + ": dimension " + to_string(d) +
                            " is out of range for a " + to_string(nd) + "-D tensor");
    return w;
}

TensorPtr permute(TensorPtr a, vector<int> dims) {
    const int nd = a->dim();
    if ((int)dims.size() != nd)
        throw runtime_error("permute: " + to_string(dims.size()) + " dims given for a " +
                            to_string(nd) + "-D tensor");
    vector<bool> seen(nd, false);
    for (int& d : dims) {
        d = wrap_dim(d, nd, "permute");
        if (seen[d]) throw runtime_error("permute: dimension repeated");
        seen[d] = true;
    }

    Shape shape(nd), strides(nd);
    for (int i = 0; i < nd; i++) {
        shape[i]   = a->shape()[dims[i]];
        strides[i] = a->strides()[dims[i]];
    }
    auto v = make_shared<Tensor>(a->storage(), shape, strides, a->offset());

    if (a->requires_grad()) {
        auto node = make_node(v, {a});
        node->backward_fn = [a, v, dims]() {
            // Walking the view in row-major order with these strides lands on
            // each element's row-major position in `a`.
            const Shape a_rm = Tensor::contiguous_strides(a->shape());
            Shape to_a(dims.size());
            for (size_t i = 0; i < dims.size(); i++) to_a[i] = a_rm[dims[i]];
            const float* gv = v->grad();
            float* ga = a->grad();
            strided::for_each(v->shape(), to_a, 0,
                              [&](int64_t i, int64_t j) { ga[j] += gv[i]; });
        };
    }
    return v;
}

TensorPtr transpose(TensorPtr a, int dim0, int dim1) {
    const int nd = a->dim();
    vector<int> dims(nd);
    for (int i = 0; i < nd; i++) dims[i] = i;
    swap(dims[wrap_dim(dim0, nd, "transpose")], dims[wrap_dim(dim1, nd, "transpose")]);
    return permute(a, dims);
}

TensorPtr reshape(TensorPtr a, ShapeArg new_shape_arg) {
    Shape shape = new_shape_arg.get();

    // at most one -1, inferred from the rest
    int infer = -1;
    int64_t known = 1;
    for (int i = 0; i < (int)shape.size(); i++) {
        if (shape[i] == -1) {
            if (infer >= 0) throw runtime_error("reshape: only one dimension can be -1");
            infer = i;
        } else {
            known *= shape[i];
        }
    }
    if (infer >= 0) {
        if (known == 0 || a->numel() % known)
            throw runtime_error("reshape: cannot infer the -1 dimension");
        shape[infer] = a->numel() / known;
        known *= shape[infer];
    }
    if (known != a->numel())
        throw runtime_error("reshape: cannot reshape " + to_string(a->numel()) +
                            " elements into a shape of " + to_string(known));

    // a non-contiguous source has no single stride pattern for the new shape,
    // so it is materialised first; the copy is part of the graph
    TensorPtr src = a->is_contiguous() ? a : contiguous(a);
    auto v = make_shared<Tensor>(src->storage(), shape, Tensor::contiguous_strides(shape),
                                 src->offset());

    if (src->requires_grad()) {
        auto node = make_node(v, {src});
        node->backward_fn = [src, v]() {
            // element order is unchanged, so the gradient maps one to one
            const int64_t n = src->numel();
            const float* gv = v->grad();
            float* gs = src->grad();
            for (int64_t i = 0; i < n; i++) gs[i] += gv[i];
        };
    }
    return v;
}

TensorPtr contiguous(TensorPtr a) {
    if (a->is_contiguous())
        return a;

    auto out = make_shared<Tensor>(a->shape(), a->dtype());
    const size_t es = dtype_size(a->dtype());
    const unsigned char* src = static_cast<const unsigned char*>(a->storage()->data());
    unsigned char* dst = static_cast<unsigned char*>(out->storage()->data());
    strided::for_each(a->shape(), a->strides(), a->offset(), [&](int64_t i, int64_t off) {
        memcpy(dst + i * es, src + off * es, es);
    });

    if (a->requires_grad()) {
        auto node = make_node(out, {a});
        node->backward_fn = [a, out]() {
            // both gradients are row-major over the same shape
            const int64_t n = a->numel();
            const float* go = out->grad();
            float* ga = a->grad();
            for (int64_t i = 0; i < n; i++) ga[i] += go[i];
        };
    }
    return out;
}

// ── Fused linear ──────────────────────────────────────────────────────────────

TensorPtr linear_op(TensorPtr x, TensorPtr W, TensorPtr bias, TensorPtr residual) {
    x = checked_input("linear_op", x);
    W = checked_input("linear_op", W);
    bias = checked_input("linear_op", bias);
    residual = checked_input("linear_op", residual);

    if (x->shape().size() != 2 || W->shape().size() != 2 || x->shape()[1] != W->shape()[0])
        throw runtime_error("linear_op: x must be [rows, in] and W [in, out]");
    const int rows = x->shape()[0], in = x->shape()[1], out = W->shape()[1];
    if (bias && bias->num_el() != out)
        throw runtime_error("linear_op: bias must have " + to_string(out) + " elements");
    if (residual && (residual->shape().size() != 2 || residual->shape()[0] != rows
                                                  || residual->shape()[1] != out))
        throw runtime_error("linear_op: residual must be [rows, out]");

    auto result = make_shared<Tensor>(Shape{rows, out});
    float* y = result->data();
    const float* rd = residual ? residual->data() : nullptr;
    const float* bd = bias ? bias->data() : nullptr;

    // seed with residual + bias, one pass, then the gemm accumulates onto it
    #pragma omp parallel for schedule(static) if((long long)rows * out >= 32768)
    for (int r = 0; r < rows; r++) {
        float* yr = y + (size_t)r * out;
        if (rd) {
            const float* rr = rd + (size_t)r * out;
            for (int j = 0; j < out; j++) yr[j] = rr[j];
        }
        if (bd)
            for (int j = 0; j < out; j++) yr[j] += bd[j];
    }
    matmul_accumulate(rows, out, in, x->data(), W->data(), y);

    vector<TensorPtr> inputs = {x, W};
    if (bias)     inputs.push_back(bias);
    if (residual) inputs.push_back(residual);
    bool need = false;
    for (auto& t : inputs) need = need || t->requires_grad();

    if (need) {
        auto node = make_node(result, inputs);
        node->backward_fn = [x, W, bias, residual, result, rows, in, out]() {
            const float* g = result->grad();

            if (x->requires_grad())
                matmul_backward_a(rows, in, out, g, W->data(), x->grad());
            if (W->requires_grad())
                matmul_backward_b(rows, in, out, g, x->data(), W->grad());

            // bias: sum over rows, parallel over the output feature so each
            // thread owns its own slot
            if (bias && bias->requires_grad()) {
                #pragma omp parallel for schedule(static) if(out >= 256)
                for (int j = 0; j < out; j++) {
                    float acc = 0.0f;
                    for (int r = 0; r < rows; r++) acc += g[(size_t)r * out + j];
                    bias->grad()[j] += acc;
                }
            }

            // residual: the gradient passes straight through
            if (residual && residual->requires_grad()) {
                const int n = rows * out;
                #pragma omp parallel for schedule(static) if(n >= 32768)
                for (int i = 0; i < n; i++) residual->grad()[i] += g[i];
            }
        };
    }
    return result;
}
