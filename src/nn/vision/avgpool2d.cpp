#include "ml/ops.hpp"
#include "ml/nn/vision/avgpool2d.hpp"

using namespace std;

AvgPool2D::AvgPool2D(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

// Parallel over (image, channel) planes; see MaxPool2D.
TensorPtr AvgPool2D::forward(TensorPtr input){
    input = checked_input("AvgPool2D", input);
    if (input->dim() != 4)
        throw runtime_error("AvgPool2D: input must be [batch, channels, height, width]");
    const int n = (int)input->shape()[0], channels = (int)input->shape()[1];
    const int h_in = (int)input->shape()[2], w_in = (int)input->shape()[3];
    const int K = kernel_size, s = stride;
    const int h_out = (h_in - K) / s + 1, w_out = (w_in - K) / s + 1;
    if (h_out <= 0 || w_out <= 0)
        throw runtime_error("AvgPool2D: window larger than the input");
    const int planes = n * channels;
    const int64_t in_hw = (int64_t)h_in * w_in, out_hw = (int64_t)h_out * w_out;
    const float inv = 1.0f / (float)(K * K);

    auto output = make_shared<Tensor>(Shape{n, channels, h_out, w_out});
    const float* X = input->data();
    float* O = output->data();

    #pragma omp parallel for schedule(static) if((int64_t)planes * out_hw * K * K >= 65536)
    for (int pl = 0; pl < planes; pl++) {
        const float* x = X + pl * in_hw;
        float* o = O + pl * out_hw;
        for (int i = 0; i < h_out; i++)
            for (int j = 0; j < w_out; j++) {
                float acc = 0.0f;
                for (int kh = 0; kh < K; kh++) {
                    const float* row = x + (int64_t)(i * s + kh) * w_in + j * s;
                    for (int kw = 0; kw < K; kw++) acc += row[kw];
                }
                o[i * w_out + j] = acc * inv;
            }
    }

    if (input->requires_grad()){
        auto node = make_node(output, {input});
        node->name = "AvgPool2D";
        node->backward_fn = [input, planes, in_hw, out_hw, h_out, w_out, w_in, K, s, inv]
                            (const TensorPtr& output) {
            const float* G = output->grad();
            float* GX = input->grad();
            #pragma omp parallel for schedule(static) if((int64_t)planes * out_hw * K * K >= 65536)
            for (int pl = 0; pl < planes; pl++) {
                float* gx = GX + pl * in_hw;
                const float* g = G + pl * out_hw;
                for (int i = 0; i < h_out; i++)
                    for (int j = 0; j < w_out; j++) {
                        const float share = g[i * w_out + j] * inv;
                        for (int kh = 0; kh < K; kh++) {
                            float* row = gx + (int64_t)(i * s + kh) * w_in + j * s;
                            for (int kw = 0; kw < K; kw++) row[kw] += share;
                        }
                    }
            }
        };
    }
    return output;
}

vector<TensorPtr> AvgPool2D::parameters(){
    return {};
}
