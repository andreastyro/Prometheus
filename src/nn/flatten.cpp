#include "ml/nn/flatten.hpp"

using namespace std;

TensorPtr Flatten::forward(TensorPtr input){

    int n = input->shape[0];
    int flat_size = input->num_el() / n;

    auto result = make_shared<Tensor>(vector<int>{n, flat_size});
    result->data = input->data;

    return result;

}

vector<TensorPtr> Flatten::parameters(){
    return {};
}
