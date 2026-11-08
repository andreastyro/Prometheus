#include "ml/optim/adamw.hpp"
#include "../ops_internal.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>

using namespace std;

AdamW::AdamW(vector<TensorPtr> params, float lr, float beta1, float beta2,
             float eps, float weight_decay)
    : lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay), t(0)
{
    this->parameters = params;
    for (auto& p : parameters) {
        m.push_back(vector<float>(p->num_el(), 0.0f));
        v.push_back(vector<float>(p->num_el(), 0.0f));
    }
}

void AdamW::step() {
    t += 1;
    const float bc1 = 1.0f - std::pow(beta1, t);
    const float bc2 = 1.0f - std::pow(beta2, t);
    const float b1 = beta1, b2 = beta2, lr_ = lr, eps_ = eps;
    // decoupled weight decay as one multiplier, applied before the adaptive
    // step: w <- w - lr*wd*w - lr*m_hat/(sqrt(v_hat)+eps). The old loop applied
    // the decay to the already-updated weight, an extra lr^2 term PyTorch's
    // AdamW does not have.
    const float decay = 1.0f - lr * weight_decay;

    for (int i = 0; i < (int)parameters.size(); i++) {
        auto& p = parameters[i];
        if (p->device() == Device::CUDA) {   // the update runs on the GPU
            const float* gd = device_grad(p);
            if (!gd) continue;
            cuda_ops::adam_step(device_weights(p), gd, m_dev.get(i, m[i]), v_dev.get(i, v[i]), p->numel(),
                                lr_, b1, b2, bc1, bc2, eps_, decay);
            continue;
        }
        const int n = (int)p->numel();
        float* w = p->data();
        const float* g = p->grad();
        float* M = m[i].data();
        float* V = v[i].data();
        #pragma omp parallel for schedule(static) if(n >= 32768)
        for (int j = 0; j < n; j++) {
            const float gj = g[j];
            M[j] = b1 * M[j] + (1.0f - b1) * gj;
            V[j] = b2 * V[j] + (1.0f - b2) * gj * gj;
            const float m_hat = M[j] / bc1;
            const float v_hat = V[j] / bc2;
            w[j] = w[j] * decay - lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

void AdamW::save_state(const string& path) const {
    for (size_t i = 0; i < m.size(); i++) {       // GPU state comes home first
        m_dev.to_host(i, const_cast<vector<float>&>(m[i]));
        v_dev.to_host(i, const_cast<vector<float>&>(v[i]));
    }
    ofstream file(path, ios::binary);
    if (!file) throw runtime_error("AdamW::save_state: cannot open " + path);

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

void AdamW::load_state(const string& path) {
    ifstream file(path, ios::binary);
    if (!file) throw runtime_error("AdamW::load_state: cannot open " + path);

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
