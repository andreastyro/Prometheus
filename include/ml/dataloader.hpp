#pragma once
#include "ml/tensor.hpp"

class DataLoader{
public:

    TensorPtr x; //input
    TensorPtr y; //output
    int batch_size;
    bool shuffle = false;

    int index;

    DataLoader(TensorPtr x, TensorPtr y, int batch_size, bool shuffle = false);

    std::pair<TensorPtr, TensorPtr> next_batch();
    bool has_next();
    void reset();

};