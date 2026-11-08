#include "ml/tensor.hpp"
#include "ml/autograd.hpp"
#include "ml/optim/adam.hpp"
#include "../ops_internal.hpp"
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>
#include <stdexcept>

using namespace std;

Adam::Adam(vector<TensorPtr> params, float lr, float beta1, float beta2, float eps){
    this->parameters = params;
    this->lr = lr;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->eps = eps;
    this->t = 0;

    for (auto& p : parameters){
        m.push_back(vector<float>(p->num_el(), 0.0f));
        v.push_back(vector<float>(p->num_el(), 0.0f));
    }
}

void Adam::step(){

    t += 1;
    float bc1 = 1.0f - std::pow(beta1, t);
    float bc2 = 1.0f - std::pow(beta2, t);

    for (int i = 0; i < (int)parameters.size(); i++){

        auto& p = parameters[i];
        if (p->device() == Device::CUDA) {   // the update runs on the GPU
            const float* gd = device_grad(p);
            if (!gd) continue;
            cuda_ops::adam_step(device_weights(p), gd, m_dev.get(i, m[i]), v_dev.get(i, v[i]), p->numel(),
                                lr, beta1, beta2, bc1, bc2, eps, 1.0f);
            continue;
        }
        const int n = p->num_el();

        // m_hat and v_hat used to be declared outside both loops, which made
        // them shared state the moment this is threaded.
        #pragma omp parallel for schedule(static) if(n >= 32768)
        for (int j = 0; j < n; j++){
            m[i][j] = beta1 * m[i][j] + (1 - beta1) * p->grad()[j];
            v[i][j] = beta2 * v[i][j] + (1 - beta2) * (p->grad()[j] * p->grad()[j]);

            const float m_hat = m[i][j] / bc1;
            const float v_hat = v[i][j] / bc2;

            p->data()[j] -= lr * m_hat / (sqrt(v_hat) + eps);
        }
    }
}

void Adam::save_state(const std::string& path) const {
    for (size_t i = 0; i < m.size(); i++) {       // GPU state comes home first
        m_dev.to_host(i, const_cast<std::vector<float>&>(m[i]));
        v_dev.to_host(i, const_cast<std::vector<float>&>(v[i]));
    }
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Adam::save_state: cannot open " + path);

    file.write(reinterpret_cast<const char*>(&t), sizeof(int));
    int n_params = (int)m.size();
    file.write(reinterpret_cast<const char*>(&n_params), sizeof(int));

    for (int i = 0; i < n_params; i++) {
        int sz = (int)m[i].size();
        file.write(reinterpret_cast<const char*>(&sz), sizeof(int));
        file.write(reinterpret_cast<const char*>(m[i].data()), sz * sizeof(float));
        file.write(reinterpret_cast<const char*>(v[i].data()), sz * sizeof(float));
    }
}

void Adam::load_state(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Adam::load_state: cannot open " + path);

    file.read(reinterpret_cast<char*>(&t), sizeof(int));
    int n_params;
    file.read(reinterpret_cast<char*>(&n_params), sizeof(int));

    m.resize(n_params);
    v.resize(n_params);
    m_dev.reset();
    v_dev.reset();
    for (int i = 0; i < n_params; i++) {
        int sz;
        file.read(reinterpret_cast<char*>(&sz), sizeof(int));
        m[i].resize(sz);
        v[i].resize(sz);
        file.read(reinterpret_cast<char*>(m[i].data()), sz * sizeof(float));
        file.read(reinterpret_cast<char*>(v[i].data()), sz * sizeof(float));
    }
}
