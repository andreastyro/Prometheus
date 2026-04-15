#include "ml/nn/conv2d.hpp"
#include <cmath>

using namespace std;

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding, string weight_init) {

    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;

    // weights: [out_channels, in_channels, kernel_size, kernel_size]
    // bias:    [out_channels]

    weights = Tensor::randn({out_channels, in_channels, kernel_size, kernel_size});
    //                   how many filters, filter depth, spatial size of filter (hxw)

    bias = Tensor::zeros({out_channels});

    int inputs_per_filter = in_channels * kernel_size * kernel_size;
    int outputs_per_filter = out_channels * kernel_size * kernel_size;

    if (weight_init == "xavier"){
        float std = sqrt(2.0f / (inputs_per_filter + outputs_per_filter));
        for (auto& v : weights->data) v *= std;
    } else if (weight_init == "kaiming"){
        float std = sqrt(2.0f / inputs_per_filter);
        for (auto& v : weights->data) v *= std;
    }

    weights->requires_grad = true;
    bias->requires_grad = true;

}
TensorPtr Conv2D::forward(TensorPtr input){
    // your implementation here

    int batch_size = input->shape[0]; 
    int in_channels = input->shape[1]; // number of input channels
    int h_in = input->shape[2]; // input height, like image or feature map
    int w_in = input->shape[3]; // input width

    int h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
    int w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

    auto output = make_shared<Tensor>(vector<int>{batch_size, out_channels, h_out, w_out});

    for (int b = 0; b < batch_size; b++){
        for (int f = 0; f < out_channels; f++){ // filter
            for (int i = 0; i < h_out; i++){ // output height
                for (int j = 0; j < w_out; j++){ // output width

                    // output[b][f][i][j] — flat index into output tensor
                    int output_idx = b * (out_channels * h_out * w_out) + f * (h_out * w_out) + i * w_out + j;

                    // accumulate dot product for one output value across all input channels and kernel positions
                    for (int ic = 0; ic < in_channels; ic++){
                        for (int kh = 0; kh < kernel_size; kh++){ // kernel height
                            for (int kw = 0; kw < kernel_size; kw++){ // kernel width
                                int src_y = i * stride + kh - padding;
                                int src_x = j * stride + kw - padding;

                                if (src_y < 0 || src_y >= h_in || src_x < 0 || src_x >= w_in) continue; // skip iteration to avoid OOB

                                // input[b][ic][src_y][src_x] — flat index into input tensor
                                int input_idx = b * (in_channels * h_in * w_in) + ic * (h_in * w_in) + src_y * w_in + src_x;

                                // weights[f][ic][kh][kw] — flat index into weights tensor
                                int weight_idx = f * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + kh * kernel_size + kw;

                                output->data[output_idx] += input->data[input_idx] * weights->data[weight_idx];
                            }
                        }
                    }
                    output->data[output_idx] += bias->data[f];
                }
            }
        }
    }

    if (input->requires_grad || weights->requires_grad) {
        
        auto node = make_node(output, {input, weights, bias});

        node->backward_fn = [=]() {
            for (int b = 0; b < batch_size; b++){
                for (int f = 0; f < out_channels; f++){
                    for (int i = 0; i < h_out; i++){
                        for (int j = 0; j < w_out; j++){

                            int output_idx = b * (out_channels * h_out * w_out) + f * (h_out * w_out) + i * w_out + j;
                            float grad = output->grad[output_idx];


                            // bias grad: one bias per filter, accumulate from every output position
                            bias->grad[f] += grad;

                            for (int ic = 0; ic < in_channels; ic++){
                                for (int kh = 0; kh < kernel_size; kh++){
                                    for (int kw = 0; kw < kernel_size; kw++){
                                        int src_y = i * stride + kh - padding;
                                        int src_x = j * stride + kw - padding;

                                        if (src_y < 0 || src_y >= h_in || src_x < 0 || src_x >= w_in) continue;

                                        int input_idx = b * (in_channels * h_in * w_in) + ic * (h_in * w_in) + src_y * w_in + src_x;
                                        int weight_idx = f * (in_channels * kernel_size * kernel_size) + ic * (kernel_size * kernel_size) + kh * kernel_size + kw;

                                        // weight grad: input value * upstream grad
                                        weights->grad[weight_idx] += input->data[input_idx] * grad;

                                        // input grad: weight value * upstream grad
                                        input->grad[input_idx] += weights->data[weight_idx] * grad;
                                    }
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

vector<TensorPtr> Conv2D::parameters(){
    return {weights, bias};
}
