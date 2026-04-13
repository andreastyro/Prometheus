#include "ml/dataloader.hpp"
#include <algorithm>
#include <numeric>
#include <memory>

using namespace std;

DataLoader::DataLoader(TensorPtr x, TensorPtr y, int batch_size, bool shuffle){

    this->x = x;
    this->y = y;
    this->batch_size = batch_size;
    this->shuffle = shuffle;

    index = 0;

    if (shuffle == true){

        int n = x->shape[0];
        vector<int> indices(n);

        for (int i = 0; i < n; i++){
            indices[i] = i;
        }

        mt19937 rng(random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng); // are we sure the values are unique

        auto new_x = make_shared<Tensor>(x->shape);
        auto new_y = make_shared<Tensor>(y->shape);

        int x_cols = x->shape[1];
        int y_cols = y->shape[1];

        for (int i = 0; i < n; i++){
            for (int j = 0; j < x_cols; j++){
                new_x->data[i * x_cols + j] = x->data[indices[i] * x_cols + j];
            }

            for (int j = 0; j < y_cols; j++){
                new_y->data[i * y_cols + j] = y->data[indices[i] * y_cols + j];
            }
        }

        this->x = new_x;
        this->y = new_y;

    }

}

void DataLoader::reset(){
    index = 0;
}

bool DataLoader::has_next(){

    return index < x->shape[0]; //inline if statement

}

pair<TensorPtr, TensorPtr> DataLoader::next_batch(){
    int start = index;
    int end = min(index + batch_size, x->shape[0]);

    int rows = end - start;

    vector<int> shape_x = {rows, x->shape[1]};
    vector<int> shape_y = {rows, y->shape[1]};

    auto batch_x = make_shared<Tensor>(shape_x);
    auto batch_y = make_shared<Tensor>(shape_y);
    
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < x->shape[1]; j++){
            batch_x->data[i * x->shape[1] + j] = x->data[(start + i) * x->shape[1] + j];
        }

        for (int j = 0; j < y->shape[1]; j++){
            batch_y->data[i * y->shape[1] + j] = y->data[(start + i) * y->shape[1] + j];
        }

    }

    index = end;

    return {batch_x, batch_y};


}