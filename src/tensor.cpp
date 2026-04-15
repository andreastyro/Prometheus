#include "ml/tensor.hpp"
#include "ml/autograd.hpp"
#include <stdio.h>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <algorithm>

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
TensorPtr Tensor::randn(vector<int> shape) {
    auto t = make_shared<Tensor>(shape);
    random_device rd;                        // seed from hardware
    mt19937 gen(rd());                       // mersenne twister generator
    normal_distribution<float> dist(0.0f, 1.0f); // mean=0, std=1
    for (float& x : t->data)
        x = dist(gen);
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

void dfs(TensorPtr node, vector<TensorPtr>& order, unordered_set<TensorPtr>& visited){
    if (visited.count(node)){
        return; // already visited
    }

    visited.insert(node);

    if (node->grad_fn) {
        for (TensorPtr& input : node->grad_fn->inputs){
            dfs(input, order, visited);
        }
    }

    order.push_back(node);
}

void Tensor::backward(){

    grad.assign(data.size(), 1.0f);

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
