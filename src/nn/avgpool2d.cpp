#include "ml/nn/avgpool2d.hpp"

using namespace std;

AvgPool2D::AvgPool2D(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

TensorPtr AvgPool2D::forward(TensorPtr input){
    int n        = input->shape[0];
    int channels = input->shape[1];
    int h_in     = input->shape[2];
    int w_in     = input->shape[3];

    int h_out = (h_in - kernel_size) / stride + 1;
    int w_out = (w_in - kernel_size) / stride + 1;
    float window_size = (float)(kernel_size * kernel_size);

    auto output = make_shared<Tensor>(vector<int>{n, channels, h_out, w_out});

    for (int b = 0; b < n; b++){
        for (int c = 0; c < channels; c++){
            for (int i = 0; i < h_out; i++){
                for (int j = 0; j < w_out; j++){

                    int output_idx = b * (channels * h_out * w_out) + c * (h_out * w_out) + i * w_out + j;
                    float sum = 0;

                    for (int kh = 0; kh < kernel_size; kh++){
                        for (int kw = 0; kw < kernel_size; kw++){
                            int src_y = i * stride + kh;
                            int src_x = j * stride + kw;
                            int input_idx = b * (channels * h_in * w_in) + c * (h_in * w_in) + src_y * w_in + src_x;
                            sum += input->data[input_idx];
                        }
                    }

                    output->data[output_idx] = sum / window_size;
                }
            }
        }
    }

    if (input->requires_grad){
        auto node = make_node(output, {input});
        node->backward_fn = [=]() { // all gradients distributed equally
            for (int b = 0; b < n; b++){
                for (int c = 0; c < channels; c++){
                    for (int i = 0; i < h_out; i++){
                        for (int j = 0; j < w_out; j++){

                            int output_idx = b * (channels * h_out * w_out) + c * (h_out * w_out) + i * w_out + j;
                            float grad = output->grad[output_idx] / window_size;

                            for (int kh = 0; kh < kernel_size; kh++){
                                for (int kw = 0; kw < kernel_size; kw++){
                                    int src_y = i * stride + kh;
                                    int src_x = j * stride + kw;
                                    int input_idx = b * (channels * h_in * w_in) + c * (h_in * w_in) + src_y * w_in + src_x;
                                    input->grad[input_idx] += grad;
                                }
                            }
                        }
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
