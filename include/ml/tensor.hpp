#pragma once
#include <vector>
#include <stdexcept>

class Tensor {
public:

    std::vector <float> data;
    std::vector <int> shape;

    Tensor(std::vector<int> shape_){
        shape = shape_;
        
        int total = 1;

        for (int i : shape) 
            total *= i;

        data.resize(total, 0.0f); // Create empty Tensor of 0s with shape_

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

    static Tensor zeros(std::vector<int> shape){
        return Tensor(shape);
    }

    static Tensor ones(std::vector<int> shape){
        Tensor t(shape);
        t.fill(1.0f);
        return t;
    }

};
