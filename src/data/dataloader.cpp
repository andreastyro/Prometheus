#include "ml/data/dataloader.hpp"
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

    if (shuffle) reshuffle();
}

void DataLoader::reset(){
    index = 0;
    if (shuffle) reshuffle();
}

void DataLoader::reshuffle(){

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

DataSplit data_split(TensorPtr x, TensorPtr y, float train_ratio, float val_ratio, float test_ratio, bool shuffle){

    int n_rows = x->shape[0];

    vector<int> indices(n_rows);
    for (int i = 0; i < n_rows; i++) indices[i] = i;

    if (shuffle){
        mt19937 rng(random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    // number of rows per split

    int train_rows = train_ratio * n_rows;
    int test_rows = test_ratio * n_rows;
    int val_rows = 0;
    if (val_ratio != 0.0f) val_rows = val_ratio * n_rows;

    DataSplit split;

    int x_cols = x->shape[1];
    int y_cols = y->shape[1];


    // train split

    split.x_train = make_shared<Tensor>(vector<int>({train_rows, x_cols}));
    split.y_train = make_shared<Tensor>(vector<int>({train_rows, y_cols}));

    for (int i = 0; i < train_rows; i++){

        for (int j = 0; j < x_cols; j++){
            split.x_train->data[i * x_cols + j] = x->data[indices[i] * x_cols + j];
        }

        for (int j = 0; j < y_cols; j++){
            split.y_train->data[i * y_cols + j] = y->data[indices[i] * y_cols + j];
        }

    }

    // val split

    if (val_ratio > 0){
        split.x_val = make_shared<Tensor>(vector<int>({val_rows, x_cols}));
        split.y_val = make_shared<Tensor>(vector<int>({val_rows, y_cols}));
    
        for (int i = 0; i < val_rows; i++){

            int src = train_rows + i;
            
            for (int j = 0; j < x_cols; j++){
                split.x_val->data[i * x_cols + j] = x->data[indices[src] * x_cols + j];
            }

            for (int j = 0; j < y_cols; j++){
                split.y_val->data[i * y_cols + j] = y->data[indices[src] * y_cols + j];
            }

        }
    }

    // test split

    split.x_test = make_shared<Tensor>(vector<int>({test_rows, x_cols}));
    split.y_test = make_shared<Tensor>(vector<int>({test_rows, y_cols}));

    for (int i = 0; i < test_rows; i++){

        int src = train_rows + val_rows + i;

        for (int j = 0; j < x_cols; j++){
            split.x_test->data[i * x_cols + j] = x->data[indices[src] * x_cols + j];
        }

        for (int j = 0; j < y_cols; j++){
            split.y_test->data[i * y_cols + j] = y->data[indices[src] * y_cols + j];
        }

    }


    return split;


}