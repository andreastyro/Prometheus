#pragma once
#include <vector>
#include <stdexcept>
#include <random>
#include <memory>
#include "ml/autograd.hpp"

class Tensor : public std::enable_shared_from_this<Tensor> {
public:

    std::vector<float> data;
    std::vector<int> shape;
    std::vector<float> grad; /**  Gradient of the loss with respect to each element in data */
    bool requires_grad = false;

    std::shared_ptr<GradNode> grad_fn = nullptr;

    // Create tensor from existing data
    Tensor(std::vector<int> shape_, std::vector<float> data_) {
        shape = shape_;
        data  = data_;
        grad.resize(data.size(), 0.0f);
    }

    Tensor(std::vector<int> shape_){
        shape = shape_;

        int total = 1;

        for (int i : shape)
            total *= i;

        data.resize(total, 0.0f); // find number of elements
        grad.resize(total, 0.0f); // number of gradients = number of weights/elements
    }

    int num_el() const{
        int total = 1;
        for (int d : shape)
            total *= d;

        return total;
    }

    float get(int row, int col){
        return data[row * shape[1] + col];
    }

    void set(int row, int col, float value){
        data[row * shape[1] + col] = value;
    }

    void print() const;

    void fill(float val){
        for(float& i : data)
            i = val;
    }

    static TensorPtr zeros(std::vector<int> shape){
        return std::make_shared<Tensor>(shape);
    }

    static TensorPtr ones(std::vector<int> shape){
        auto t = std::make_shared<Tensor>(shape);
        t->fill(1.0f);
        return t;
    }

    void reset_grad() {
        grad.assign(data.size(), 0.0f);
    }

    // Random normal values (mean=0, std=1)
    static TensorPtr randn(std::vector<int> shape);

    // Flip rows and cols (2D only)
    TensorPtr transpose() const;

    // Change shape without changing data
    TensorPtr reshape(std::vector<int> new_shape) const;

    // Backpropagate gradients through the computation graph
    void backward();

};
