#include "ml/nn/vision/convtranspose2d.hpp"
#include "ml/nn/vision/im2col.hpp"
#include "ml/matmul_backend.hpp"
#include "ml/ops.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
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
        for (auto& v : weights->values()) v *= std;
    } else if (weight_init == "kaiming"){
        float std = sqrt(2.0f / fan_in);
        for (auto& v : weights->values()) v *= std;
    }

    weights->set_requires_grad(true);
    bias->set_requires_grad(true);
}

// A transposed convolution is the adjoint of a convolution and reuses the same
// machinery the other way round: per image, one gemm  col = W^T @ x  builds
// the columns and col2im scatters them onto the larger output. Its backward is
// a convolution: im2col of the output gradient and two gemms.
//
// This replaced nested scalar scatter loops that measured 288x slower than
// PyTorch at [16, 64, 16, 16] -> 32 channels, k4 s2.
TensorPtr ConvTranspose2D::forward(TensorPtr input){
    if (input->device() != Device::CUDA) input = checked_input("ConvTranspose2D", input);
    if (input->dim() != 4)
        throw runtime_error("ConvTranspose2D: input must be [batch, channels, height, width]");
    const int B = (int)input->shape()[0], Cin = (int)input->shape()[1];
    const int H = (int)input->shape()[2], W = (int)input->shape()[3];
    if (Cin != in_channels)
        throw runtime_error("ConvTranspose2D: expected " + to_string(in_channels) +
                            " input channels, got " + to_string(Cin));
    const int K = kernel_size, Cout = out_channels, s = stride, p = padding;
    const int Ho = (H - 1) * s - 2 * p + K, Wo = (W - 1) * s - 2 * p + K;
    if (Ho <= 0 || Wo <= 0)
        throw runtime_error("ConvTranspose2D: padding leaves no output");
    const int HW = H * W, OHW = Ho * Wo, CKK = Cout * K * K;
    const int64_t in_sz = (int64_t)Cin * HW, out_sz = (int64_t)Cout * OHW;

    // The forward written with ops: the GPU path, and the reference the
    // higher-order gradients differentiate.
    auto reference = [B, Cin, H, W, K, Cout, s, p, Ho, Wo, HW, CKK](const std::vector<TensorPtr>& in) -> TensorPtr {
            // the output image is what a convolution would read patches from
            const int CC = Cout, IH = Ho, IW = Wo, OH = H, OW = W;
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
            auto col = matmul(transpose(reshape(in[1], {(int64_t)Cin, (int64_t)CKK})),
                              reshape(in[0], {(int64_t)B, (int64_t)Cin, (int64_t)HW}));      // [B, CKK, HW]
            auto out = scatter_add_flat(col, idx, {(int64_t)B, (int64_t)Cout, (int64_t)Ho, (int64_t)Wo}); // col2im
            return add(out, reshape(in[2], {1, (int64_t)Cout, 1, 1}));
        };
    if (input->device() == Device::CUDA || weights->device() == Device::CUDA)
        return reference({input, weights, bias});

    auto output = make_shared<Tensor>(Shape{B, Cout, Ho, Wo});
    const float* X = input->data();
    const float* Wt = weights->data();
    const float* bv = bias->data();
    float* O = output->data();

    // W^T [CKK, Cin], built once so the per-image gemm is untransposed
    vector<float> WT((size_t)CKK * Cin);
    for (int c = 0; c < Cin; c++)
        for (int k = 0; k < CKK; k++) WT[(size_t)k * Cin + c] = Wt[(size_t)c * CKK + k];

    vision::ImageLoop loop(B);
    vector<vector<float>> cols(loop.slots());
    loop.run([&](int b, int slot) {
        auto& col = cols[slot];
        col.resize((size_t)CKK * HW);
        // col [CKK, HW] = W^T @ x_b, with W [Cin, CKK]
        gemm_serial(false, false, CKK, HW, Cin, WT.data(), Cin, X + b * in_sz, HW, 0.0f, col.data(), HW);
        float* ob = O + b * out_sz;
        for (int c = 0; c < Cout; c++) std::fill(ob + (int64_t)c * OHW, ob + (int64_t)(c + 1) * OHW, bv[c]);
        vision::col2im_add(col.data(), Cout, Ho, Wo, K, s, p, H, W, ob);
    });

    if (input->requires_grad() || weights->requires_grad() || bias->requires_grad()) {
        auto node = make_node(output, {input, weights, bias});
        node->name = "ConvTranspose2D";
        autograd::attach_reference(*node, reference);
        auto w = weights;
        auto bvec = bias;
        node->backward_fn = [input, w, bvec, B, Cin, H, W, K, s, p, Ho, Wo, Cout, HW, OHW, CKK, in_sz, out_sz]
                            (const TensorPtr& output) {
            const float* G = output->grad();
            const float* X = input->data();
            const float* Wt = w->data();
            const bool need_x = input->requires_grad(), need_w = w->requires_grad();
            float* GX = need_x ? input->grad() : nullptr;

            vision::ImageLoop loop(B);
            vector<vector<float>> dw(loop.slots()), db(loop.slots());
            vector<vector<float>> scratch(loop.slots());
            loop.run([&](int b, int slot) {
                const float* gb = G + b * out_sz;                  // [Cout, Ho, Wo]
                if (bvec->requires_grad()) {
                    auto& d = db[slot];
                    if (d.empty()) d.assign(Cout, 0.0f);
                    for (int c = 0; c < Cout; c++) {
                        float acc = 0.0f;
                        for (int i = 0; i < OHW; i++) acc += gb[(int64_t)c * OHW + i];
                        d[c] += acc;
                    }
                }
                if (!need_x && !need_w) return;
                // the output gradient as columns: the layout the forward built
                auto& gcol = scratch[slot];
                gcol.resize((size_t)CKK * HW);
                vision::im2col(gb, Cout, Ho, Wo, K, s, p, H, W, gcol.data());
                if (need_x)   // dx_b [Cin, HW] += W [Cin, CKK] @ gcol
                    gemm_serial(false, false, Cin, HW, CKK, Wt, CKK, gcol.data(), HW, 1.0f,
                                GX + b * in_sz, HW);
                if (need_w) { // dW [Cin, CKK] += x_b [Cin, HW] @ gcol^T
                    auto& d = dw[slot];
                    if (d.empty()) d.assign((size_t)Cin * CKK, 0.0f);
                    gemm_serial(false, true, Cin, CKK, HW, X + b * in_sz, HW, gcol.data(), HW, 1.0f,
                                d.data(), CKK);
                }
            });
            if (need_w) vision::reduce_partials(dw, w->grad(), (int64_t)Cin * CKK);
            if (bvec->requires_grad()) vision::reduce_partials(db, bvec->grad(), Cout);
        };
    }
    return output;
}

vector<TensorPtr> ConvTranspose2D::parameters(){
    return {weights, bias};
}
