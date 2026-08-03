#include "ml/optim/adamw.hpp"
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
    float bc1 = 1.0f - std::pow(beta1, t);
    float bc2 = 1.0f - std::pow(beta2, t);

    for (int i = 0; i < (int)parameters.size(); i++) {
        auto& p = parameters[i];
        for (int j = 0; j < p->num_el(); j++) {
            float g = p->grad[j];

            m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * g;
            v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * g * g;

            float m_hat = m[i][j] / bc1;
            float v_hat = v[i][j] / bc2;

            // adaptive update (same as Adam)
            p->data[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            // decoupled weight decay (NOT folded into gradient)
            p->data[j] -= lr * weight_decay * p->data[j];
        }
    }
}

void AdamW::save_state(const string& path) const {
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
    for (int i = 0; i < n_params; i++) {
        int sz;
        file.read(reinterpret_cast<char*>(&sz), sizeof(int));
        m[i].resize(sz);
        v[i].resize(sz);
        file.read(reinterpret_cast<char*>(m[i].data()), sz * sizeof(float));
        file.read(reinterpret_cast<char*>(v[i].data()), sz * sizeof(float));
    }
}
