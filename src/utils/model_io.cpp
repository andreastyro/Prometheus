#include "ml/utils/model_io.hpp"
#include <fstream>

using namespace std;

void save(const string& path, vector<TensorPtr> params){
    ofstream file(path, ios::binary);

    int num_params = params.size();
    file.write((char*)&num_params, sizeof(int));

    for (auto& p : params){
        int ndims = p->shape.size();
        file.write((char*)&ndims, sizeof(int));

        for (int dim : p->shape)
            file.write((char*)&dim, sizeof(int));

        int n = p->num_el();
        file.write((char*)p->data.data(), n * sizeof(float));
    }
}

vector<TensorPtr> load(const string& path){
    ifstream file(path, ios::binary);

    int num_params;
    file.read((char*)&num_params, sizeof(int));

    vector<TensorPtr> params;

    for (int i = 0; i < num_params; i++){
        int ndims;
        file.read((char*)&ndims, sizeof(int));

        vector<int> shape(ndims);
        for (int d = 0; d < ndims; d++)
            file.read((char*)&shape[d], sizeof(int));

        auto tensor = make_shared<Tensor>(shape);
        file.read((char*)tensor->data.data(), tensor->num_el() * sizeof(float));

        params.push_back(tensor);
    }

    return params;
}

