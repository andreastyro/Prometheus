#include "ml/nn/convtranspose2d.hpp"
#include <cmath>

using namespace std;

ConvTranspose2D::ConvTranspose2D(int in_channels, int out_channels, int kernel_size, int stride, int padding, string weight_init)
{
    this->in_channels  = in_channels;
    this->out_channels = out_channels;
    this->kernel_size  = kernel_size;
    this->stride       = stride;
    this->padding      = padding;

    // weights: [in_channels, out_channels, kernel_size, kernel_size]
    // note: transposed — in/out swapped vs Conv2D
    weights = Tensor::randn({in_channels, out_channels, kernel_size, kernel_size});
    bias    = Tensor::zeros({out_channels});

    int fan_in  = in_channels  * kernel_size * kernel_size;
    int fan_out = out_channels * kernel_size * kernel_size;

    if (weight_init == "xavier"){
        float std = sqrt(2.0f / (fan_in + fan_out));
        for (auto& v : weights->data) v *= std;
    } else if (weight_init == "kaiming"){
        float std = sqrt(2.0f / fan_in);
        for (auto& v : weights->data) v *= std;
    }

    weights->requires_grad = true;
    bias->requires_grad    = true;
}

TensorPtr ConvTranspose2D::forward(TensorPtr input){
    int batch_size = input->shape[0];
    int h_in       = input->shape[2];
    int w_in       = input->shape[3];

    // output spatial size
    int h_out = (h_in - 1) * stride - 2 * padding + kernel_size;
    int w_out = (w_in - 1) * stride - 2 * padding + kernel_size;

    auto output = make_shared<Tensor>(vector<int>{batch_size, out_channels, h_out, w_out});

    // ConvTranspose2D forward: for each input position, scatter-add into the output
    // Each input value at (b, ic, i, j) contributes to a kernel-sized region of the output
    for (int b = 0; b < batch_size; b++){
        for (int ic = 0; ic < in_channels; ic++){
            for (int i = 0; i < h_in; i++){
                for (int j = 0; j < w_in; j++){

                    int input_idx = b * (in_channels * h_in * w_in) + ic * (h_in * w_in) + i * w_in + j;
                    float val = input->data[input_idx];

                    for (int oc = 0; oc < out_channels; oc++){
                        for (int kh = 0; kh < kernel_size; kh++){
                            for (int kw = 0; kw < kernel_size; kw++){
                                int out_y = i * stride + kh - padding;
                                int out_x = j * stride + kw - padding;

                                if (out_y < 0 || out_y >= h_out || out_x < 0 || out_x >= w_out) continue;

                                int weight_idx = ic * (out_channels * kernel_size * kernel_size) + oc * (kernel_size * kernel_size) + kh * kernel_size + kw;
                                int output_idx = b * (out_channels * h_out * w_out) + oc * (h_out * w_out) + out_y * w_out + out_x;

                                output->data[output_idx] += val * weights->data[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // add bias to each output channel
    for (int b = 0; b < batch_size; b++)
        for (int oc = 0; oc < out_channels; oc++)
            for (int i = 0; i < h_out; i++)
                for (int j = 0; j < w_out; j++)
                    output->data[b * (out_channels * h_out * w_out) + oc * (h_out * w_out) + i * w_out + j] += bias->data[oc];

    if (input->requires_grad || weights->requires_grad){
        auto node = make_node(output, {input, weights, bias});

        node->backward_fn = [=]() {
            for (int b = 0; b < batch_size; b++){
                for (int ic = 0; ic < in_channels; ic++){
                    for (int i = 0; i < h_in; i++){
                        for (int j = 0; j < w_in; j++){

                            int input_idx = b * (in_channels * h_in * w_in) + ic * (h_in * w_in) + i * w_in + j;
                            float grad_acc = 0;

                            for (int oc = 0; oc < out_channels; oc++){
                                for (int kh = 0; kh < kernel_size; kh++){
                                    for (int kw = 0; kw < kernel_size; kw++){
                                        int out_y = i * stride + kh - padding;
                                        int out_x = j * stride + kw - padding;

                                        if (out_y < 0 || out_y >= h_out || out_x < 0 || out_x >= w_out) continue;

                                        int weight_idx = ic * (out_channels * kernel_size * kernel_size) + oc * (kernel_size * kernel_size) + kh * kernel_size + kw;
                                        int output_idx = b * (out_channels * h_out * w_out) + oc * (h_out * w_out) + out_y * w_out + out_x;

                                        float g = output->grad[output_idx];

                                        // weight grad
                                        weights->grad[weight_idx] += input->data[input_idx] * g;

                                        // input grad
                                        grad_acc += weights->data[weight_idx] * g;
                                    }
                                }
                            }
                            input->grad[input_idx] += grad_acc;
                        }
                    }
                }
            }

            // bias grad
            for (int b = 0; b < batch_size; b++)
                for (int oc = 0; oc < out_channels; oc++)
                    for (int i = 0; i < h_out; i++)
                        for (int j = 0; j < w_out; j++)
                            bias->grad[oc] += output->grad[b * (out_channels * h_out * w_out) + oc * (h_out * w_out) + i * w_out + j];
        };
    }

    return output;
}

vector<TensorPtr> ConvTranspose2D::parameters(){
    return {weights, bias};
}
