#include "ml/nn/flatten.hpp"
#include "ml/tensor.hpp"

using namespace std;

// Flatten only reinterprets the shape: element i of the output is element i of
// the input, so the gradient passes straight through unchanged.
//
// This used to copy input->data into a fresh tensor and return it with no
// grad_fn, which severed the graph. In the usual CNN arrangement
// (Conv2D -> Flatten -> Linear) that left every convolution upstream with a
// gradient of exactly zero while the Linear head trained normally.
TensorPtr Flatten::forward(TensorPtr input){

    int n = input->shape()[0];
    int flat_size = input->num_el() / n;

    auto result = make_shared<Tensor>(Shape{n, flat_size});
    result->set_data(input->to_vector());

    if (input->requires_grad()) {
        auto node = make_node(result, {input});

        node->backward_fn = [input, result]() {
            for (int i = 0; i < result->num_el(); i++)
                input->grad()[i] += result->grad()[i];
        };
    }

    return result;
}

vector<TensorPtr> Flatten::parameters(){
    return {};
}
