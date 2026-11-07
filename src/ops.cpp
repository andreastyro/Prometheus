#include "ml/tensor.hpp"
#include "ml/autograd.hpp"
#include "ml/matmul_backend.hpp"
#include <cmath>
#include <string>

using namespace std;

// A strided view owns no `data`. Every op below reads memory directly, so a
// view reaching one of them would read an empty buffer and crash. Until the
// ops learn to walk strides, they refuse views with a message that says what
// to do instead. The view ops themselves (transpose_view, reshape_view,
// contiguous) are the only ones that accept a view, and are exempt.
static void reject_view(const char* op, const TensorPtr& t) {
    if (t && t->is_view())
        throw runtime_error(string(op) + ": was handed a strided view, which "
            "owns no data of its own. Pass it through contiguous() first, or "
            "use matmul_bt if the goal is to multiply by a transpose.");
}



// Element-wise addition: a + b
TensorPtr add(TensorPtr a, TensorPtr b) {
    reject_view("add", a);
    reject_view("add", b);
    if (a->shape == b->shape){

        auto result = make_shared<Tensor>(a->shape);

        for(int i = 0; i < a->num_el(); i++){
            result->data[i] = a->data[i] + b->data[i];
        }

        if (a->requires_grad || b->requires_grad) {
            auto node = make_node(result, {a, b});

            node->backward_fn = [a, b, result](){
                for (int i = 0; i < result->num_el(); i++){
                    if (a->requires_grad) a->grad[i] += result->grad[i];
                    if (b->requires_grad) b->grad[i] += result->grad[i];
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
    reject_view("add", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++)
        result->data[i] = scalar + a->data[i];

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            for (int i = 0; i < result->num_el(); i++)
                a->grad[i] += result->grad[i];
        };
    }

    return result;
}

// Element-wise multiplication: a * b
TensorPtr multiply(TensorPtr a, TensorPtr b) {
    reject_view("multiply", a);
    reject_view("multiply", b);

    if (a->shape == b->shape){

        auto result = make_shared<Tensor>(a->shape);

        for(int i = 0; i < a->num_el(); i++){
            result->data[i] = a->data[i] * b->data[i];
        }

        if (a->requires_grad || b->requires_grad) {
            auto node = make_node(result, {a, b});
            node->backward_fn = [a, b, result](){
                for (int i = 0; i < result->num_el(); i++){
                    if (a->requires_grad) a->grad[i] += b->data[i] * result->grad[i];
                    if (b->requires_grad) b->grad[i] += a->data[i] * result->grad[i];
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
    reject_view("multiply", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++)
        result->data[i] = scalar * a->data[i];

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, scalar]() {
            for (int i = 0; i < result->num_el(); i++)
                a->grad[i] += scalar * result->grad[i];
        };
    }

    return result;
}

// Matrix multiplication: a x b
TensorPtr matmul(TensorPtr a, TensorPtr b) {
    reject_view("matmul", a);
    reject_view("matmul", b);

    if (a->shape[1] == b->shape[0]){

        auto result = make_shared<Tensor>(vector<int>{a->shape[0], b->shape[1]});

        matmul_forward(a->shape[0], b->shape[1], a->shape[1],
                       a->data.data(), b->data.data(), result->data.data());

        if (a->requires_grad || b->requires_grad){
            auto node = make_node(result, {a, b});

            node->backward_fn = [a, b, result](){
                int a_rows = a->shape[0], inner = a->shape[1], b_cols = b->shape[1];

                if (a->requires_grad)
                    matmul_backward_a(a_rows, inner, b_cols,
                                      result->grad.data(), b->data.data(), a->grad.data());

                if (b->requires_grad)
                    matmul_backward_b(a_rows, inner, b_cols,
                                      result->grad.data(), a->data.data(), b->grad.data());
            };
        }

        return result;

    } else if (b->shape[1] == a->shape[0]) {
        // user passed them in the wrong order, swap and retry
        return matmul(b, a);

    } else {
        throw runtime_error("matmul: a.cols must equal b.rows, got " + to_string(a->shape[1]) + " and " + to_string(b->shape[0]));
    }
}

// ReLU activation: max(0, x) for each element
TensorPtr relu(TensorPtr a) {
    reject_view("relu", a);

    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++){
        result->data[i] = a->data[i] > 0 ? a->data[i] : 0.0f;
    }

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            for (int i = 0; i < result->num_el(); i++)
                if (a->data[i] > 0) a->grad[i] += result->grad[i];
        };
    }

    return result;
}

// Sigmoid activation: 1 / (1 + e^-x) for each element
TensorPtr sigmoid(TensorPtr a) {
    reject_view("sigmoid", a);

    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++){
        result->data[i] = 1.0f / (1.0f + exp(-a->data[i]));
    }

    if (a->requires_grad){
        auto node = make_node(result, {a});

        node->backward_fn = [a, result](){
            for (int i = 0; i < result->num_el(); i++){
                a->grad[i] += result->data[i] * (1 - result->data[i]) * result->grad[i];
            }
        };
    }

    return result;
}

// Element-wise subtraction: a - b
TensorPtr subtract(TensorPtr a, TensorPtr b) {
    reject_view("subtract", a);
    reject_view("subtract", b);

    if (a->shape == b->shape){

        auto difference = make_shared<Tensor>(a->shape);

        for (int i = 0; i < a->num_el(); i++){
            difference->data[i] = a->data[i] - b->data[i];
        }

        if (a->requires_grad || b->requires_grad) {
            auto node = make_node(difference, {a, b});
            node->backward_fn = [a, b, difference]() {
                for (int i = 0; i < difference->num_el(); i++) {
                    if (a->requires_grad) a->grad[i] += difference->grad[i];
                    if (b->requires_grad) b->grad[i] -= difference->grad[i];
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
    reject_view("subtract", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++)
        result->data[i] = scalar - a->data[i];

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            for (int i = 0; i < result->num_el(); i++)
                a->grad[i] -= result->grad[i];
        };
    }

    return result;
}

// Tensor - scalar  (d/da = 1)
TensorPtr subtract(TensorPtr a, float scalar) {
    reject_view("subtract", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++)
        result->data[i] = a->data[i] - scalar;

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            for (int i = 0; i < result->num_el(); i++)
                a->grad[i] += result->grad[i];
        };
    }

    return result;
}

// Element-wise division: a / b
TensorPtr divide(TensorPtr a, TensorPtr b) {
    reject_view("divide", a);
    reject_view("divide", b);
    if (a->shape == b->shape){

        auto result = make_shared<Tensor>(a->shape);

        for (int i = 0; i < a->num_el(); i++){

            if (b->data[i] == 0){
                int row = i / a->shape[1];
                int col = i % a->shape[1];
                throw runtime_error("Division by zero at row " + to_string(row) + ", col " + to_string(col));
            }

            result->data[i] = a->data[i] / b->data[i];
        }

        if (a->requires_grad || b->requires_grad){

            auto node = make_node(result, {a, b});

            node->backward_fn = [a, b, result](){
                for (int i = 0; i < result->num_el(); i++){
                    if (a->requires_grad) a->grad[i] += result->grad[i] / b->data[i];
                    if (b->requires_grad) b->grad[i] += -result->grad[i] * a->data[i] / (b->data[i] * b->data[i]);
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
    reject_view("divide", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++) {
        if (a->data[i] == 0)
            throw std::runtime_error("Division by zero at index " + std::to_string(i));
        result->data[i] = scalar / a->data[i];
    }

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, scalar]() {
            for (int i = 0; i < result->num_el(); i++)
                a->grad[i] += -scalar * result->grad[i] / (a->data[i] * a->data[i]);
        };
    }

    return result;
}

// Tensor / scalar  dL/da = result.grad / scalar
TensorPtr divide(TensorPtr a, float scalar) {
    reject_view("divide", a);
    if (scalar == 0)
        throw std::runtime_error("Division by zero");
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++)
        result->data[i] = a->data[i] / scalar;

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, scalar]() {
            for (int i = 0; i < result->num_el(); i++)
                a->grad[i] += result->grad[i] / scalar;
        };
    }

    return result;
}

// Tanh activation: (e^x - e^-x) / (e^x + e^-x)
TensorPtr tanh_op(TensorPtr a) {
    reject_view("tanh_op", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++){
        result->data[i] = (exp(a->data[i]) - exp(-a->data[i])) / (exp(a->data[i]) + exp(-a->data[i]));
    }

    if (a->requires_grad){
        auto node = make_node(result, {a});

        node->backward_fn = [a, result](){
            for (int i = 0; i < result->num_el(); i++){
                a->grad[i] += (1 - result->data[i] * result->data[i]) * result->grad[i];
            }
        };
    }

    return result;
}

// Softmax: converts logits to probabilities along each row
TensorPtr softmax(TensorPtr a) {
    reject_view("softmax", a);

    auto result = make_shared<Tensor>(a->shape);

    // Subtract the row maximum before exponentiating. softmax(x + c) equals
    // softmax(x) for any constant c, so this changes nothing mathematically
    // while keeping every exponent at or below zero.
    //
    // Without it exp() overflows to inf and the result is NaN: measured here,
    // a row whose largest logit was 90 already produced NaN, and attention
    // scores or an untrained output layer reach that easily. The backward pass
    // is unaffected, since it is written in terms of the output probabilities.
    const int cols = a->shape[1];

    for (int i = 0; i < a->shape[0]; i++){

        float row_max = a->data[i * cols];
        for (int j = 1; j < cols; j++)
            if (a->data[i * cols + j] > row_max)
                row_max = a->data[i * cols + j];

        float s = 0;

        for (int j = 0; j < cols; j++){
            result->data[i * cols + j] = exp(a->data[i * cols + j] - row_max);
            s += result->data[i * cols + j];
        }

        for (int j = 0; j < cols; j++){
            result->data[i * cols + j] = result->data[i * cols + j] / s;
        }
    }

    if (a->requires_grad){

        auto node = make_node(result, {a});
        int rows = a->shape[0], cols = a->shape[1];

        node->backward_fn = [a, result, cols, rows](){

            Tensor jacobian({cols, cols});

            for (int i = 0; i < rows; i++){

                // fill jacobian for row i
                for (int j = 0; j < cols; j++){
                    for (int k = 0; k < cols; k++){
                        float s_j = result->data[i * cols + j];
                        float s_k = result->data[i * cols + k];

                        jacobian.data[j * cols + k] = s_j * ((j == k ? 1.0f : 0.0f) - s_k);
                    }
                }

                for (int j = 0; j < cols; j++){
                    for (int k = 0; k < cols; k++){
                        a->grad[i * cols + j] += jacobian.data[j * cols + k] * result->grad[i * cols + k];
                    }
                }
            }
        };
    }

    return result;
}

// Element-wise natural log
TensorPtr log_op(TensorPtr a) {
    reject_view("log_op", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++){
        if (a->data[i] <= 0)
            throw runtime_error("Log of non-positive value at index " + to_string(i));
        result->data[i] = log(a->data[i]);
    }

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            for (int i = 0; i < result->num_el(); i++)
                a->grad[i] += result->grad[i] / a->data[i];  // d(ln x)/dx = 1/x
        };
    }

    return result;
}

// Element-wise e^x
TensorPtr exp_op(TensorPtr a) {
    reject_view("exp_op", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++){
        result->data[i] = exp(a->data[i]);
    }

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            for (int i = 0; i < result->num_el(); i++)
                a->grad[i] += result->data[i] * result->grad[i];  // d(e^x)/dx = e^x = result
        };
    }

    return result;
}

// Element-wise power: x^p
TensorPtr pow_op(TensorPtr a, float p) {
    reject_view("pow_op", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++){
        result->data[i] = pow(a->data[i], p);
    }

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, p]() {
            for (int i = 0; i < result->num_el(); i++)
                a->grad[i] += p * pow(a->data[i], p - 1) * result->grad[i];  // d(x^p)/dx = p*x^(p-1)
        };
    }

    return result;
}

// Element-wise square root
TensorPtr sqrt_op(TensorPtr a) {
    reject_view("sqrt_op", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++){
        if (a->data[i] < 0)
            throw runtime_error("Cannot compute square root of negative value " +
                                to_string(a->data[i]) + " at index " + to_string(i));
        result->data[i] = sqrt(a->data[i]);
    }

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            for (int i = 0; i < result->num_el(); i++)
                a->grad[i] += result->grad[i] / (2.0f * result->data[i]);  // d(sqrt(x))/dx = 1/(2*sqrt(x))
        };
    }

    return result;
}

// Element-wise absolute value
TensorPtr abs_op(TensorPtr a) {
    reject_view("abs_op", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++){
        result->data[i] = abs(a->data[i]);
    }

    if (a->requires_grad){
        auto node = make_node(result, {a});

        node->backward_fn = [a, result](){
            for (int i = 0; i < result->num_el(); i++){
                if (a->data[i] > 0) a->grad[i] += result->grad[i];
                else if (a->data[i] < 0) a->grad[i] -= result->grad[i];
            }
        };
    }

    return result;
}

// Sum all elements (axis=-1) or along a given axis
TensorPtr sum(TensorPtr a, int axis) {
    reject_view("sum", a);

    // A 1-D tensor has no shape[1]. Reading it unconditionally was an
    // out-of-bounds vector access: harmless-looking in release builds, where it
    // returns whatever follows the shape vector, and caught immediately by a
    // debug standard library. mean(x, -1) and sum(x, -1) are legitimate on a
    // 1-D tensor and must not need a column count at all.
    const int rows = a->shape[0];
    const int cols = (a->shape.size() > 1) ? a->shape[1] : 1;
    if (axis != -1 && a->shape.size() < 2)
        throw runtime_error("axis " + to_string(axis) +
                            " requires a 2-D tensor; this tensor is 1-D");

    // global sum — returns a single value
    if (axis == -1) {
        auto result = make_shared<Tensor>(vector<int>{1});
        for (int i = 0; i < a->num_el(); i++)
            result->data[0] += a->data[i];

        if (a->requires_grad) {
            auto node = make_node(result, {a});
            node->backward_fn = [a, result]() {
                for (int i = 0; i < a->num_el(); i++)
                    a->grad[i] += result->grad[0];  // every element gets the same gradient
            };
        }
        return result;

    // sum down rows — one sum per column e.g. {2,3} -> {3}
    } else if (axis == 0) {
        auto result = make_shared<Tensor>(vector<int>{cols});
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result->data[j] += a->data[i * cols + j];

        if (a->requires_grad) {
            auto node = make_node(result, {a});
            node->backward_fn = [a, result, rows, cols]() {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a->grad[i * cols + j] += result->grad[j];  // each element gets its column's gradient
            };
        }
        return result;

    // sum across columns — one sum per row e.g. {2,3} -> {2}
    } else if (axis == 1) {
        auto result = make_shared<Tensor>(vector<int>{rows});
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result->data[i] += a->data[i * cols + j];

        if (a->requires_grad) {
            auto node = make_node(result, {a});
            node->backward_fn = [a, result, rows, cols]() {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a->grad[i * cols + j] += result->grad[i];  // each element gets its row's gradient
            };
        }
        return result;

    } else {
        throw runtime_error("Invalid axis " + to_string(axis) + ", must be -1, 0, or 1");
    }
}

// Mean of all elements (axis=-1) or along a given axis
TensorPtr mean(TensorPtr a, int axis) {
    reject_view("mean", a);

    // A 1-D tensor has no shape[1]. Reading it unconditionally was an
    // out-of-bounds vector access: harmless-looking in release builds, where it
    // returns whatever follows the shape vector, and caught immediately by a
    // debug standard library. mean(x, -1) and sum(x, -1) are legitimate on a
    // 1-D tensor and must not need a column count at all.
    const int rows = a->shape[0];
    const int cols = (a->shape.size() > 1) ? a->shape[1] : 1;
    if (axis != -1 && a->shape.size() < 2)
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
        result = make_shared<Tensor>(vector<int>{1});
        for (int i = 0; i < a->num_el(); i++)
            result->data[0] += a->data[i];
        result->data[0] /= count;
    } else if (axis == 0) {
        result = make_shared<Tensor>(vector<int>{cols});
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result->data[j] += a->data[i * cols + j];
        for (int j = 0; j < cols; j++)
            result->data[j] /= count;
    } else {
        result = make_shared<Tensor>(vector<int>{rows});
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result->data[i] += a->data[i * cols + j];
        for (int i = 0; i < rows; i++)
            result->data[i] /= count;
    }

    if (a->requires_grad) {
        auto node = make_node(result, {a});

        node->backward_fn = [a, result, axis, count, rows, cols]() {
            if (axis == -1) {
                for (int i = 0; i < a->num_el(); i++)
                    a->grad[i] += result->grad[0] / count;

            } else if (axis == 0) {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a->grad[i * cols + j] += result->grad[j] / count;

            } else if (axis == 1) {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        a->grad[i * cols + j] += result->grad[i] / count;
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
    reject_view("max_op", a);
    auto result = make_shared<Tensor>(vector<int>{1});
    result->data[0] = a->data[0];
    int arg = 0;
    for (int i = 1; i < a->num_el(); i++)
        if (a->data[i] > result->data[0]) {
            result->data[0] = a->data[i];
            arg = i;
        }

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, arg]() {
            a->grad[arg] += result->grad[0];
        };
    }

    return result;
}

// Min element — returns a scalar tensor
// Min element — mirror of max_op, gradient reaches only the arg-min.
TensorPtr min_op(TensorPtr a) {
    reject_view("min_op", a);
    auto result = make_shared<Tensor>(vector<int>{1});
    result->data[0] = a->data[0];
    int arg = 0;
    for (int i = 1; i < a->num_el(); i++)
        if (a->data[i] < result->data[0]) {
            result->data[0] = a->data[i];
            arg = i;
        }

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, arg]() {
            a->grad[arg] += result->grad[0];
        };
    }

    return result;
}

// Clip — clamp all values between min and max
TensorPtr clip(TensorPtr a, float min_val, float max_val) {
    reject_view("clip", a);
    auto result = make_shared<Tensor>(a->shape);
    for (int i = 0; i < a->num_el(); i++){
        if (a->data[i] > max_val){
            result->data[i] = max_val;
        } else if (a->data[i] < min_val) {
            result->data[i] = min_val;
        } else {
            result->data[i] = a->data[i];
        }
    }

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result, min_val, max_val]() {
            for (int i = 0; i < result->num_el(); i++) {
                // gradient is 1 where value wasn't clamped, 0 where it was
                if (a->data[i] > min_val && a->data[i] < max_val)
                    a->grad[i] += result->grad[i];
            }
        };
    }

    return result;
}

// Broadcast add — adds a 1D bias tensor to every row of a 2D tensor
// e.g. a={32,64} + b={64} -> {32,64}
TensorPtr broadcast_add(TensorPtr a, TensorPtr b) {
    reject_view("broadcast_add", a);
    reject_view("broadcast_add", b);

    if (a->shape.size() != 2)
        throw runtime_error("broadcast_add: 'a' must be a 2D tensor, got " + to_string(a->shape.size()) + "D");

    if (b->shape.size() != 1)
        throw runtime_error("broadcast_add: 'b' must be a 1D tensor, got " + to_string(b->shape.size()) + "D");

    if (b->shape[0] != a->shape[1])
        throw runtime_error("broadcast_add: 'b' size " + to_string(b->shape[0]) +
                            " must match a's columns " + to_string(a->shape[1]));

    auto result = make_shared<Tensor>(a->shape);

    for (int i = 0; i < a->shape[0]; i++){
        for (int j = 0; j < a->shape[1]; j++){
            result->data[i * a->shape[1] + j] = a->data[i * a->shape[1] + j] + b->data[j];
        }
    }

    if (a->requires_grad || b->requires_grad) {
        auto node = make_node(result, {a, b});
        node->backward_fn = [a, b, result]() {
            int rows = a->shape[0], cols = a->shape[1];

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (a->requires_grad)
                        a->grad[i * cols + j] += result->grad[i * cols + j];

                    if (b->requires_grad)
                        b->grad[j] += result->grad[i * cols + j];  // accumulate over all rows
                }
            }
        };
    }

    return result;
}

// Argmax — returns a 1D tensor of class indices, one per row
TensorPtr argmax(TensorPtr a) {
    reject_view("argmax", a);
    int rows = a->shape[0];
    int cols = a->shape[1];
    auto result = make_shared<Tensor>(vector<int>{rows});

    for (int i = 0; i < rows; i++){
        int best = 0;
        float max_val = a->data[i * cols];
        for (int j = 1; j < cols; j++){
            if (a->data[i * cols + j] > max_val){
                max_val = a->data[i * cols + j];
                best = j;
            }
        }
        result->data[i] = (float)best;
    }

    return result;
}

// GELU activation — tanh approximation used by GPT-2/3/4.
// Formula: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
TensorPtr gelu(TensorPtr a) {
    reject_view("gelu", a);
    auto result = make_shared<Tensor>(a->shape);
    int n = a->num_el();
    constexpr float kSqrt2OverPi = 0.7978845608f;

    for (int i = 0; i < n; i++) {
        float x = a->data[i];
        float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
        result->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            constexpr float k = 0.7978845608f;
            int m = a->num_el();
            for (int i = 0; i < m; i++) {
                float x = a->data[i];
                float inner = k * (x + 0.044715f * x * x * x);
                float t = tanhf(inner);
                float d_inner = k * (1.0f + 3.0f * 0.044715f * x * x);
                float grad = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * d_inner;
                a->grad[i] += result->grad[i] * grad;
            }
        };
    }

    return result;
}

// reshape_op — gradient-tracking reshape (unlike Tensor::reshape which is in-place).
TensorPtr reshape_op(TensorPtr a, std::vector<int> new_shape) {
    reject_view("reshape_op", a);
    auto result = make_shared<Tensor>(new_shape);
    result->data = a->data;

    if (a->requires_grad) {
        auto node = make_node(result, {a});
        node->backward_fn = [a, result]() {
            for (int i = 0; i < (int)result->grad.size(); i++)
                a->grad[i] += result->grad[i];
        };
    }

    return result;
}

TensorPtr transpose_op(TensorPtr a) {
    reject_view("transpose_op", a);
    if (a->shape.size() != 2)
        throw runtime_error("transpose_op: only supported for 2D tensors");

    int rows = a->shape[0], cols = a->shape[1];
    auto result = make_shared<Tensor>(vector<int>{cols, rows});

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result->data[j * rows + i] = a->data[i * cols + j];

    if (a->requires_grad) {
        auto node = make_node(result, {a});

        // grad_a[i, j] += grad_result[j, i]
        node->backward_fn = [a, result, rows, cols]() {
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    a->grad[i * cols + j] += result->grad[j * rows + i];
        };
    }

    return result;
}

TensorPtr concat_op(vector<TensorPtr> inputs, int axis) {
    for (auto& t_ : inputs) reject_view("concat_op", t_);
    if (inputs.empty())
        throw runtime_error("concat_op: no inputs");

    const vector<int> shape0 = inputs[0]->shape;
    const int ndim = (int)shape0.size();
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim)
        throw runtime_error("concat_op: axis out of range");

    // every input must match on all axes except the concatenation one
    int concat_total = 0;
    for (auto& t : inputs) {
        if ((int)t->shape.size() != ndim)
            throw runtime_error("concat_op: inputs differ in rank");
        for (int d = 0; d < ndim; d++)
            if (d != axis && t->shape[d] != shape0[d])
                throw runtime_error("concat_op: inputs differ on axis " + to_string(d));
        concat_total += t->shape[axis];
    }

    vector<int> out_shape = shape0;
    out_shape[axis] = concat_total;
    auto result = make_shared<Tensor>(out_shape);

    // Treat the tensor as [outer, axis_dim, inner]: copying a contiguous
    // inner-sized run per outer index per input is all that is needed.
    int outer = 1;
    for (int d = 0; d < axis; d++) outer *= shape0[d];
    int inner = 1;
    for (int d = axis + 1; d < ndim; d++) inner *= shape0[d];

    bool any_grad = false;
    for (auto& t : inputs) if (t->requires_grad) any_grad = true;

    for (int o = 0; o < outer; o++) {
        int written = 0;
        for (auto& t : inputs) {
            const int span = t->shape[axis] * inner;
            const float* src = t->data.data() + o * span;
            float* dst = result->data.data() + o * concat_total * inner + written;
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
                    const int span = t->shape[axis] * inner;
                    if (t->requires_grad) {
                        const float* src = result->grad.data() + o * concat_total * inner + read;
                        float* dst = t->grad.data() + o * span;
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
    reject_view("matmul_bt", a);
    reject_view("matmul_bt", b);
    if (a->shape.size() != 2 || b->shape.size() != 2)
        throw runtime_error("matmul_bt: both operands must be 2-D");

    const int m = a->shape[0], k = a->shape[1];
    const int n = b->shape[0];
    if (b->shape[1] != k)
        throw runtime_error("matmul_bt: a.cols must equal b.cols, got "
                            + to_string(k) + " and " + to_string(b->shape[1]));

    auto result = make_shared<Tensor>(vector<int>{m, n});
    matmul_forward_bt(m, n, k, a->data.data(), b->data.data(),
                      result->data.data());

    if (a->requires_grad || b->requires_grad) {
        auto node = make_node(result, {a, b});

        node->backward_fn = [a, b, result, m, n, k]() {
            const float* g = result->grad.data();

            // C = A @ B^T  =>  dA = dC @ B,  dB = dC^T @ A
            if (a->requires_grad)
                matmul_accumulate(m, k, n, g, b->data.data(), a->grad.data());

            if (b->requires_grad)
                matmul_backward_b(m, n, k, a->data.data(), g, b->grad.data());
        };
    }

    return result;
}

// ── Views ────────────────────────────────────────────────────────────────────
//
// See the gradient convention in tensor.hpp: a view's grad is row-major over
// the view's own shape. That is what makes the backward passes below simple --
// contiguous() is a straight element-for-element accumulate, and the only one
// that rearranges anything is the transpose, which rearranges exactly as the
// forward did.

TensorPtr transpose_view(TensorPtr a) {
    if (a->shape.size() != 2)
        throw runtime_error("transpose_view: only 2-D tensors are supported");

    auto base = a->is_view() ? a->base : a;
    auto in_strides = a->is_view() ? a->strides
                                   : Tensor::contiguous_strides(a->shape);

    const int rows = a->shape[0], cols = a->shape[1];

    auto v = make_shared<Tensor>();         // no storage allocated
    v->shape   = {cols, rows};
    v->base    = base;
    v->offset  = a->offset;
    v->strides = {in_strides[1], in_strides[0]};

    if (a->requires_grad) {
        auto node = make_node(v, {a}, true);
        node->backward_fn = [a, v, rows, cols]() {
            // v->grad is row-major over [cols, rows]; a->grad over [rows, cols]
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    a->grad[i * cols + j] += v->grad[j * rows + i];
        };
    }
    return v;
}

TensorPtr reshape_view(TensorPtr a, vector<int> new_shape) {
    if (!a->is_contiguous())
        throw runtime_error("reshape_view: the input is a non-contiguous view; "
                            "pass it through contiguous() first");

    int total = 1;
    for (int d : new_shape) total *= d;
    if (total != a->num_el())
        throw runtime_error("reshape_view: cannot reshape " + to_string(a->num_el())
                            + " elements into a shape of " + to_string(total));

    auto base = a->is_view() ? a->base : a;
    auto v = make_shared<Tensor>();         // no storage allocated
    v->shape   = new_shape;
    v->base    = base;
    v->offset  = a->offset;
    v->strides = Tensor::contiguous_strides(new_shape);

    if (a->requires_grad) {
        auto node = make_node(v, {a}, true);
        node->backward_fn = [a, v]() {
            // element order is unchanged, so this is one to one
            const int n = a->num_el();
            for (int i = 0; i < n; i++)
                a->grad[i] += v->grad[i];
        };
    }
    return v;
}

TensorPtr contiguous(TensorPtr a) {
    if (!a->is_view())
        return a;

    const int n  = a->num_el();
    const int nd = (int)a->shape.size();

    auto out = make_shared<Tensor>(a->shape);

    // gather in row-major order of the shape, stepping through the strides
    vector<int> idx(nd, 0);
    for (int i = 0; i < n; i++) {
        out->data[i] = a->view_at(idx);
        for (int d = nd - 1; d >= 0; d--) {
            if (++idx[d] < a->shape[d]) break;
            idx[d] = 0;
        }
    }

    if (a->requires_grad) {
        auto node = make_node(out, {a}, true);
        node->backward_fn = [a, out, n]() {
            // Both gradients are row-major over the same shape, so this is a
            // straight accumulate. Walking the strides here was the defect in
            // the first attempt: it wrote the view's gradient in storage order
            // while transpose_view read it in row-major order.
            a->ensure_grad();
            for (int i = 0; i < n; i++)
                a->grad[i] += out->grad[i];
        };
    }
    return out;
}
