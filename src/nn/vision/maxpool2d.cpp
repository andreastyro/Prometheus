#include "ml/ops.hpp"
#include "ml/nn/vision/maxpool2d.hpp"
#include <limits>

using namespace std;

MaxPool2D::MaxPool2D(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

// Parallel over (image, channel) planes. A plane's windows only read that
// plane and its argmax indices only point into it, so the backward scatter is
// race-free per plane even when windows overlap (stride < kernel).
TensorPtr MaxPool2D::forward(TensorPtr input){
    input = checked_input("MaxPool2D", input);
    if (input->dim() != 4)
        throw runtime_error("MaxPool2D: input must be [batch, channels, height, width]");
    const int n = (int)input->shape()[0], channels = (int)input->shape()[1];
    const int h_in = (int)input->shape()[2], w_in = (int)input->shape()[3];
    const int K = kernel_size, s = stride;
    const int h_out = (h_in - K) / s + 1, w_out = (w_in - K) / s + 1;
    if (h_out <= 0 || w_out <= 0)
        throw runtime_error("MaxPool2D: window larger than the input");
    const int planes = n * channels;
    const int64_t in_hw = (int64_t)h_in * w_in, out_hw = (int64_t)h_out * w_out;

    auto output = make_shared<Tensor>(Shape{n, channels, h_out, w_out});
    // argmax of each window as an offset within its plane, for the backward
    auto mask = make_shared<vector<int>>((size_t)output->numel());
    const float* X = input->data();
    float* O = output->data();
    int* M = mask->data();

    #pragma omp parallel for schedule(static) if((int64_t)planes * out_hw * K * K >= 65536)
    for (int pl = 0; pl < planes; pl++) {
        const float* x = X + pl * in_hw;
        float* o = O + pl * out_hw;
        int* m = M + pl * out_hw;
        for (int i = 0; i < h_out; i++)
            for (int j = 0; j < w_out; j++) {
                int best = (i * s) * w_in + j * s;
                float best_v = x[best];
                for (int kh = 0; kh < K; kh++) {
                    const float* row = x + (int64_t)(i * s + kh) * w_in + j * s;
                    for (int kw = 0; kw < K; kw++)
                        if (row[kw] > best_v) { best_v = row[kw]; best = (i * s + kh) * w_in + j * s + kw; }
                }
                o[i * w_out + j] = best_v;
                m[i * w_out + j] = best;
            }
    }

    if (input->requires_grad()){
        auto node = make_node(output, {input});
        node->name = "MaxPool2D";
        node->backward_fn = [input, mask, planes, in_hw, out_hw](const TensorPtr& output) {
            const float* G = output->grad();
            float* GX = input->grad();
            const int* M = mask->data();
            #pragma omp parallel for schedule(static) if((int64_t)planes * out_hw >= 65536)
            for (int pl = 0; pl < planes; pl++) {
                float* gx = GX + pl * in_hw;
                const float* g = G + pl * out_hw;
                const int* m = M + pl * out_hw;
                for (int64_t k = 0; k < out_hw; k++) gx[m[k]] += g[k];
            }
        };
    }
    return output;
}

vector<TensorPtr> MaxPool2D::parameters(){
    return {};
}
