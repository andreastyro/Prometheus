#include "ml/nn/maxpool2d.hpp"
#include <limits>

using namespace std;

MaxPool2D::MaxPool2D(int kernel_size, int stride)
    : kernel_size(kernel_size), stride(stride == -1 ? kernel_size : stride) {}

TensorPtr MaxPool2D::forward(TensorPtr input){
    int n = input->shape[0];
    int channels = input->shape[1];
    int h_in = input->shape[2];
    int w_in = input->shape[3];

    int h_out = (h_in - kernel_size) / stride + 1;
    int w_out = (w_in - kernel_size) / stride + 1;

    auto output = make_shared<Tensor>(vector<int>{n, channels, h_out, w_out});

    // store the index of the max value for each output position (needed for backward)
    vector<int> mask(output->num_el());

    for (int b = 0; b < n; b++){
        for (int c = 0; c < channels; c++){
            for (int i = 0; i < h_out; i++){
                for (int j = 0; j < w_out; j++){

                    int output_idx = b * (channels * h_out * w_out) + c * (h_out * w_out) + i * w_out + j;

                    float max_val = -numeric_limits<float>::infinity();
                    int max_idx = -1;

                    for (int kh = 0; kh < kernel_size; kh++){
                        for (int kw = 0; kw < kernel_size; kw++){
                            int src_y = i * stride + kh;
                            int src_x = j * stride + kw;

                            int input_idx = b * (channels * h_in * w_in) + c * (h_in * w_in) + src_y * w_in + src_x;
                            if (input->data[input_idx] > max_val){
                                max_val = input->data[input_idx];
                                max_idx = input_idx;
                            }
                        }
                    }

                    output->data[output_idx] = max_val;
                    mask[output_idx] = max_idx;
                }
            }
        }
    }

    if (input->requires_grad){ // simply add grad to max value as its the only one that modifies output
        auto node = make_node(output, {input});
        node->backward_fn = [input, output, mask]() {
            for (int i = 0; i < output->num_el(); i++)
                input->grad[mask[i]] += output->grad[i];
        };
    }

    return output;
}

vector<TensorPtr> MaxPool2D::parameters(){
    return {};
}
