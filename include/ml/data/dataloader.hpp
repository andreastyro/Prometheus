#pragma once
#include "ml/tensor.hpp"


struct DataSplit {
    TensorPtr x_train, y_train;
    TensorPtr x_val, y_val;
    TensorPtr x_test, y_test; 

    // 3 assignments to show pairs
};

class DataLoader{
public:

    TensorPtr x; //input
    TensorPtr y; //output
    int batch_size;
    bool shuffle = false;

    int index;

    DataLoader(TensorPtr x, TensorPtr y, int batch_size, bool shuffle = false);

    std::pair<TensorPtr, TensorPtr> next_batch(); // return next batch
    bool has_next(); // check if next batch exists
    void reset(); // reset index to 0 for next epoch
    void reshuffle(); // reshuflle after each epoch

};

DataSplit data_split(TensorPtr x, TensorPtr y, float train_ratio, float test_ratio, float val_ratio = 0.0f, bool shuffle = false);