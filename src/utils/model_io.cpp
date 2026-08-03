#include "ml/utils/model_io.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>

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

// ── Checkpoint ────────────────────────────────────────────────────────────────

static const char CKPT_MAGIC[4] = {'C', 'K', 'P', 'T'};
static const int  CKPT_VERSION  = 1;

void save_checkpoint(const string& path, const vector<TensorPtr>& params,
                     int epoch, float loss) {
    ofstream file(path, ios::binary);
    if (!file) throw runtime_error("save_checkpoint: cannot open " + path);

    file.write(CKPT_MAGIC, 4);
    file.write(reinterpret_cast<const char*>(&CKPT_VERSION), sizeof(int));
    file.write(reinterpret_cast<const char*>(&epoch),        sizeof(int));
    file.write(reinterpret_cast<const char*>(&loss),         sizeof(float));

    int num_params = (int)params.size();
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(int));

    for (auto& p : params) {
        int ndims = (int)p->shape.size();
        file.write(reinterpret_cast<const char*>(&ndims), sizeof(int));
        for (int dim : p->shape)
            file.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        int n = p->num_el();
        file.write(reinterpret_cast<const char*>(p->data.data()), n * sizeof(float));
    }
}

Checkpoint load_checkpoint(const string& path, vector<TensorPtr>& params) {
    ifstream file(path, ios::binary);
    if (!file) throw runtime_error("load_checkpoint: cannot open " + path);

    char magic[4];
    file.read(magic, 4);
    if (memcmp(magic, CKPT_MAGIC, 4) != 0)
        throw runtime_error("load_checkpoint: not a valid checkpoint file");

    int version;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));

    Checkpoint ckpt;
    file.read(reinterpret_cast<char*>(&ckpt.epoch), sizeof(int));
    file.read(reinterpret_cast<char*>(&ckpt.loss),  sizeof(float));

    int num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(int));

    params.resize(num_params);
    for (int i = 0; i < num_params; i++) {
        int ndims;
        file.read(reinterpret_cast<char*>(&ndims), sizeof(int));
        vector<int> shape(ndims);
        for (int d = 0; d < ndims; d++)
            file.read(reinterpret_cast<char*>(&shape[d]), sizeof(int));
        auto tensor = make_shared<Tensor>(shape);
        file.read(reinterpret_cast<char*>(tensor->data.data()), tensor->num_el() * sizeof(float));
        params[i] = tensor;
    }

    return ckpt;
}

