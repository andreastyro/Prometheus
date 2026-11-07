#include "ml/tensor.hpp"
#include "ml/autograd.hpp"
#include <stdio.h>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <cstring>   // memcpy

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
// One generator for the whole library, so results are reproducible after
// manual_seed(). Previously randn() built a fresh mt19937 from random_device on
// every call, which made every run differ and every weight init unrepeatable.
static mt19937& global_rng() {
    static mt19937 gen(random_device{}());
    return gen;
}

mt19937& prometheus_rng() { return global_rng(); }

bool &grad_enabled() {
    static bool enabled = true;
    return enabled;
}

void Tensor::manual_seed(unsigned int seed) {
    global_rng().seed(seed);
}

TensorPtr Tensor::randn(vector<int> shape) {
    auto t = make_shared<Tensor>(shape);
    normal_distribution<float> dist(0.0f, 1.0f); // mean=0, std=1
    for (float& x : t->data)
        x = dist(global_rng());
    return t;
}

// Flip rows and cols (2D only)
TensorPtr Tensor::transpose() const {

    if (shape.size() != 2)
        throw runtime_error("transpose: only supported for 2D tensors");

    int rows = shape[0], cols = shape[1];
    auto result = make_shared<Tensor>(vector<int>{cols, rows});
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result->data[j * rows + i] = data[i * cols + j];
    return result;
}

// Change shape without changing data
TensorPtr Tensor::reshape(vector<int> new_shape) const {
    int new_total = 1;
    for (int i : new_shape)
        new_total *= i;

    if (new_total != num_el())
        throw runtime_error("reshape: cannot reshape tensor of size " + to_string(num_el()) +
                            " into shape with size " + to_string(new_total));

    auto result = make_shared<Tensor>(new_shape);
    result->data = data;
    return result;
}

// Iterative post-order traversal.
//
// This was recursive, which puts one C++ stack frame per graph node: a long
// sequence or a deep residual stack is thousands of nodes and can overflow the
// stack with no useful diagnostic. The explicit stack has no such limit.
void dfs(TensorPtr node, vector<TensorPtr>& order, unordered_set<TensorPtr>& visited){
    vector<pair<TensorPtr, bool>> stack;      // (node, children_expanded)
    stack.push_back({node, false});

    while (!stack.empty()) {
        auto [cur, expanded] = stack.back();
        stack.pop_back();

        if (expanded) {
            order.push_back(cur);
            continue;
        }
        if (!cur || visited.count(cur)) continue;
        visited.insert(cur);

        stack.push_back({cur, true});         // emit after its inputs
        if (cur->grad_fn)
            for (auto& input : cur->grad_fn->inputs)
                if (input && !visited.count(input))
                    stack.push_back({input, false});
    }
}

void Tensor::backward(){

    // A tensor with no grad_fn has nothing behind it. Walking the graph would
    // silently do nothing and leave every parameter at zero, which is the same
    // failure mode as a severed graph and just as hard to notice -- the usual
    // cause is forgetting to leave a NoGrad scope.
    if (!grad_fn)
        throw runtime_error(
            "backward() called on a tensor with no computation graph. It is "
            "either a leaf, or it was produced while gradients were disabled "
            "(NoGrad / no_grad()).");

    grad.assign((size_t)num_el(), 1.0f);

    vector<TensorPtr> order;
    unordered_set<TensorPtr> visited;

    dfs(shared_from_this(), order, visited); // order of nodes

    reverse(order.begin(), order.end());

    for (TensorPtr& t : order){
        if (t->grad_fn){
            t->grad_fn->backward_fn();
        }
    }

}

TensorPtr Tensor::detach() const {
    auto t = make_shared<Tensor>(shape, data);
    t->requires_grad = false;
    return t;
}

// float32 bits -> float16 bits (IEEE 754 half)
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint16_t sign  = (x >> 16) & 0x8000;
    int32_t  exp   = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant  = x & 0x7FFFFF;
    if (exp <= 0)  return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

// float16 bits -> float32
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t x;
    if (exp == 0)       x = sign | (mant << 13);
    else if (exp == 31) x = sign | 0x7F800000 | (mant << 13);
    else                x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
}

TensorPtr Tensor::half() const {
    auto out = make_shared<Tensor>(shape);
    for (int i = 0; i < (int)data.size(); i++)
        out->data[i] = f16_to_f32(f32_to_f16(data[i]));
    return out;
}

TensorPtr Tensor::to_float() const {
    return make_shared<Tensor>(shape, data);
}
