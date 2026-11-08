#include "ml/nn/vision/conv2d.hpp"
#include "ml/nn/vision/im2col.hpp"
#include "ml/matmul_backend.hpp"
#include "ml/ops.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
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
        for (auto& v : weights->values()) v *= std;
    } else if (weight_init == "kaiming"){
        float std = sqrt(2.0f / inputs_per_filter);
        for (auto& v : weights->values()) v *= std;
    }

    weights->set_requires_grad(true);
    bias->set_requires_grad(true);

}
// Convolution as im2col + gemm, the way PyTorch runs it on a CPU without
// oneDNN: per image, the patches are unrolled into a [C*K*K, Ho*Wo] column
// matrix and one gemm  out = W[F, C*K*K] @ col  produces every output
// channel. The gemm goes to MKL / OpenBLAS. Images run in parallel, each with
// a single-threaded gemm and its column matrix in cache (see ImageLoop).
//
// This replaced seven nested scalar loops on one core: 316x slower than
// PyTorch at [16, 32, 32, 32] -> 64 channels, and a backward that wrote the
// bias gradient from inside its innermost loop.
TensorPtr Conv2D::forward(TensorPtr input){
    if (input->device() != Device::CUDA) input = checked_input("Conv2D", input);
    if (input->dim() != 4)
        throw runtime_error("Conv2D: input must be [batch, channels, height, width]");
    const int B = (int)input->shape()[0], C = (int)input->shape()[1];
    const int H = (int)input->shape()[2], W = (int)input->shape()[3];
    if (C != in_channels)
        throw runtime_error("Conv2D: expected " + to_string(in_channels) + " input channels, got " +
                            to_string(C));
    const int K = kernel_size, F = out_channels, s = stride, p = padding;
    const int Ho = (H + 2 * p - K) / s + 1, Wo = (W + 2 * p - K) / s + 1;
    if (Ho <= 0 || Wo <= 0)
        throw runtime_error("Conv2D: kernel larger than the padded input");
    const int HW = Ho * Wo, CKK = C * K * K;
    const int64_t in_sz = (int64_t)C * H * W, out_sz = (int64_t)F * HW;

    // The forward written with ops: the GPU path, and the reference the
    // higher-order gradients differentiate.
    auto reference = [B, C, H, W, K, F, s, p, Ho, Wo, HW, CKK](const std::vector<TensorPtr>& in) -> TensorPtr {
            const int CC = C, IH = H, IW = W, OH = Ho, OW = Wo;
            // the patch of every output position, as flat indices into the input;
            // -1 marks padding, which gathers as 0
            auto idx = make_shared<vector<int64_t>>((size_t)B * CKK * HW);
            for (int b = 0; b < B; b++)
                for (int c = 0; c < CC; c++)
                    for (int kh = 0; kh < K; kh++)
                        for (int kw = 0; kw < K; kw++) {
                            const int64_t row = ((int64_t)b * CKK + (c * K + kh) * K + kw) * HW;
                            for (int i = 0; i < OH; i++)
                                for (int j = 0; j < OW; j++) {
                                    const int y = i * s + kh - p, x = j * s + kw - p;
                                    (*idx)[row + i * OW + j] = (y < 0 || y >= IH || x < 0 || x >= IW)
                                        ? -1 : (((int64_t)b * CC + c) * IH + y) * IW + x;
                                }
                        }
            auto col = gather_flat(in[0], idx, {(int64_t)B, (int64_t)CKK, (int64_t)HW});   // im2col
            auto out = matmul(reshape(in[1], {(int64_t)F, (int64_t)CKK}), col);            // [B, F, HW]
            out = add(out, reshape(in[2], {1, (int64_t)F, 1}));
            return reshape(out, {(int64_t)B, (int64_t)F, (int64_t)Ho, (int64_t)Wo});
        };
    if (input->device() == Device::CUDA || weights->device() == Device::CUDA)
        return reference({input, weights, bias});

    auto output = make_shared<Tensor>(Shape{B, F, Ho, Wo});
    const float* X = input->data();
    const float* Wt = weights->data();
    const float* bv = bias->data();
    float* O = output->data();

    vision::ImageLoop loop(B);
    vector<vector<float>> cols(loop.slots());
    loop.run([&](int b, int slot) {
        auto& col = cols[slot];
        col.resize((size_t)CKK * HW);
        vision::im2col(X + b * in_sz, C, H, W, K, s, p, Ho, Wo, col.data());
        float* ob = O + b * out_sz;
        for (int f = 0; f < F; f++) std::fill(ob + (int64_t)f * HW, ob + (int64_t)(f + 1) * HW, bv[f]);
        gemm_serial(false, false, F, HW, CKK, Wt, CKK, col.data(), HW, 1.0f, ob, HW);
    });

    if (input->requires_grad() || weights->requires_grad() || bias->requires_grad()) {
        auto node = make_node(output, {input, weights, bias});
        node->name = "Conv2D";
        autograd::attach_reference(*node, reference);
        auto w = weights;
        auto bvec = bias;
        node->backward_fn = [input, w, bvec, B, C, H, W, K, s, p, Ho, Wo, F, HW, CKK, in_sz, out_sz]
                            (const TensorPtr& output) {
            const float* G = output->grad();
            const float* X = input->data();
            const float* Wt = w->data();
            const bool need_x = input->requires_grad(), need_w = w->requires_grad();
            float* GX = need_x ? input->grad() : nullptr;

            // W^T [CKK, F], built once: a plain gemm on it ran faster than a
            // transposed-A gemm on W, and it is only F*CKK floats
            vector<float> WT;
            if (need_x) {
                WT.resize((size_t)CKK * F);
                for (int f = 0; f < F; f++)
                    for (int k = 0; k < CKK; k++) WT[(size_t)k * F + f] = Wt[(size_t)f * CKK + k];
            }

            vision::ImageLoop loop(B);
            // dW and db sum over every image, so each thread keeps its own
            // and they are added at the end
            vector<vector<float>> dw(loop.slots()), db(loop.slots());
            vector<vector<float>> scratch(loop.slots());
            loop.run([&](int b, int slot) {
                const float* gb = G + b * out_sz;                  // [F, HW]
                auto& col = scratch[slot];
                col.resize((size_t)CKK * HW);
                if (bvec->requires_grad()) {
                    auto& d = db[slot];
                    if (d.empty()) d.assign(F, 0.0f);
                    for (int f = 0; f < F; f++) {
                        float acc = 0.0f;
                        for (int i = 0; i < HW; i++) acc += gb[(int64_t)f * HW + i];
                        d[f] += acc;
                    }
                }
                if (need_w) {
                    // dW [F, CKK] += g [F, HW] @ col^T
                    auto& d = dw[slot];
                    if (d.empty()) d.assign((size_t)F * CKK, 0.0f);
                    vision::im2col(X + b * in_sz, C, H, W, K, s, p, Ho, Wo, col.data());
                    gemm_serial(false, true, F, CKK, HW, gb, HW, col.data(), HW, 1.0f, d.data(), CKK);
                }
                if (need_x) {
                    // dcol [CKK, HW] = W^T @ g, scattered back onto the image
                    gemm_serial(false, false, CKK, HW, F, WT.data(), F, gb, HW, 0.0f, col.data(), HW);
                    vision::col2im_add(col.data(), C, H, W, K, s, p, Ho, Wo, GX + b * in_sz);
                }
            });
            if (need_w) vision::reduce_partials(dw, w->grad(), (int64_t)F * CKK);
            if (bvec->requires_grad()) vision::reduce_partials(db, bvec->grad(), F);
        };
    }
    return output;
}

vector<TensorPtr> Conv2D::parameters(){
    return {weights, bias};
}
