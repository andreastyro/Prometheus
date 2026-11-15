// GPU kernels for convolution (our im2col kernels, or cuDNN), pooling,
// BatchNorm and GroupNorm. Split from cuda_ops.cpp; shared helpers in
// cuda_ops_common.hpp.
#include "cuda_ops_common.hpp"
#include "ml/autograd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

using namespace std;
using namespace cuda_common;

namespace cuda_ops {

// ── convolution and pooling on the device ────────────────────────────────────
//
// The standard non-cuDNN route: im2col turns every output position's patch
// into a column, so a convolution is one batched gemm per call; col2im is its
// adjoint (a gather here, so no atomics and a deterministic sum). Pooling keeps
// each window's winner on the device. Gradients: kernels for a plain backward,
// the layer's op-form reference when the gradient is itself recorded.

static const string& VISION_SRC() {
    static const string s = src(R"(
// col[b, (c*K + kh)*K + kw, oy*Wc + ox] = x[b, c, oy*s + kh - p, ox*s + kw - p] (0 outside)
extern "C" __global__ void v_im2col(const float* x, float* col, long long B, long long C, long long H, long long W,
                                    int K, int s, int p, long long Hc, long long Wc) {
    const long long HW = Hc * Wc, CKK = C * K * K;
    long long i = gid(); if (i >= B * CKK * HW) return;
    const long long pos = i % HW, r = (i / HW) % CKK, b = i / (HW * CKK);
    const long long oy = pos / Wc, ox = pos % Wc;
    const int kw = (int)(r % K), kh = (int)((r / K) % K);
    const long long c = r / (K * K);
    const long long y = oy * s + kh - p, xx = ox * s + kw - p;
    col[i] = (y < 0 || y >= H || xx < 0 || xx >= W) ? 0.0f : x[((b * C + c) * H + y) * W + xx];
}
// the adjoint: x[b, c, y, xx] = sum over the (kh, kw) whose window covers it
extern "C" __global__ void v_col2im(const float* col, float* x, long long B, long long C, long long H, long long W,
                                    int K, int s, int p, long long Hc, long long Wc) {
    long long i = gid(); if (i >= B * C * H * W) return;
    const long long xx = i % W, y = (i / W) % H, c = (i / (W * H)) % C, b = i / (W * H * C);
    const long long HW = Hc * Wc, CKK = C * K * K;
    float acc = 0.0f;
    for (int kh = 0; kh < K; kh++) {
        const long long ty = y + p - kh;
        if (ty < 0 || ty % s) continue;
        const long long oy = ty / s;
        if (oy >= Hc) continue;
        for (int kw = 0; kw < K; kw++) {
            const long long tx = xx + p - kw;
            if (tx < 0 || tx % s) continue;
            const long long ox = tx / s;
            if (ox >= Wc) continue;
            acc += col[(b * CKK + (c * K + kh) * K + kw) * HW + oy * Wc + ox];
        }
    }
    x[i] = acc;
}
// The same two with the batch folded into the columns: col[r, b*HW + pos], a
// [C*K*K, B*HW] matrix. Then a convolution over the whole batch is one gemm
// rather than one per image.
extern "C" __global__ void v_im2col_b(const float* x, float* col, long long B, long long C, long long H, long long W,
                                      int K, int s, int p, long long Hc, long long Wc) {
    const long long HW = Hc * Wc, CKK = C * K * K, BHW = B * HW;
    long long i = gid(); if (i >= CKK * BHW) return;
    const long long r = i / BHW, q = i % BHW, b = q / HW, pos = q % HW;
    const long long oy = pos / Wc, ox = pos % Wc;
    const int kw = (int)(r % K), kh = (int)((r / K) % K);
    const long long c = r / (K * K);
    const long long y = oy * s + kh - p, xx = ox * s + kw - p;
    col[i] = (y < 0 || y >= H || xx < 0 || xx >= W) ? 0.0f : x[((b * C + c) * H + y) * W + xx];
}
extern "C" __global__ void v_col2im_b(const float* col, float* x, long long B, long long C, long long H, long long W,
                                      int K, int s, int p, long long Hc, long long Wc) {
    long long i = gid(); if (i >= B * C * H * W) return;
    const long long xx = i % W, y = (i / W) % H, c = (i / (W * H)) % C, b = i / (W * H * C);
    const long long HW = Hc * Wc, BHW = B * HW;
    float acc = 0.0f;
    for (int kh = 0; kh < K; kh++) {
        const long long ty = y + p - kh;
        if (ty < 0 || ty % s) continue;
        const long long oy = ty / s;
        if (oy >= Hc) continue;
        for (int kw = 0; kw < K; kw++) {
            const long long tx = xx + p - kw;
            if (tx < 0 || tx % s) continue;
            const long long ox = tx / s;
            if (ox >= Wc) continue;
            acc += col[((c * K + kh) * K + kw) * BHW + b * HW + oy * Wc + ox];
        }
    }
    x[i] = acc;
}
// [C, B*P] -> [B, C, P], adding bias[c] if given; and [B, C, P] -> [C, B*P]
extern "C" __global__ void v_from_cb(const float* y, const float* bias, float* out, long long B, long long C, long long P) {
    long long i = gid(); if (i >= B * C * P) return;
    const long long pos = i % P, c = (i / P) % C, b = i / (P * C);
    out[i] = y[c * B * P + b * P + pos] + (bias ? bias[c] : 0.0f);
}
extern "C" __global__ void v_to_cb(const float* x, float* out, long long B, long long C, long long P) {
    long long i = gid(); if (i >= B * C * P) return;
    const long long q = i % (B * P), c = i / (B * P), b = q / P, pos = q % P;
    out[i] = x[(b * C + c) * P + pos];
}
// y[b, c, pos] += bias[c], in place
extern "C" __global__ void v_add_bias_c(float* y, const float* bias, long long C, long long P, long long n) {
    long long i = gid(); if (i >= n) return;
    y[i] += bias[(i / P) % C];
}
// max pooling: value and the flat input index of each window's winner (first max)
extern "C" __global__ void v_maxpool(const float* x, float* y, long long* arg, long long planes, long long H, long long W,
                                     int K, int s, long long Ho, long long Wo) {
    long long i = gid(); if (i >= planes * Ho * Wo) return;
    const long long ox = i % Wo, oy = (i / Wo) % Ho, pl = i / (Wo * Ho);
    const float* xp = x + pl * H * W;
    long long best = (oy * s) * W + ox * s;
    float bv = xp[best];
    for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) {
            const long long j = (oy * s + kh) * W + ox * s + kw;
            if (xp[j] > bv) { bv = xp[j]; best = j; }
        }
    y[i] = bv;
    arg[i] = pl * H * W + best;
}
extern "C" __global__ void v_scatter_rows(const float* g, const long long* idx, float* dx, long long n) {
    long long i = gid(); if (i >= n) return;
    atomicAdd(&dx[idx[i]], g[i]);
}
// max-pool backward when windows do not overlap (stride >= K): each input
// element is in at most one window, so it reads that window's gradient if it
// was the winner and writes 0 otherwise -- every element written once, with
// no zero-fill and no atomics
extern "C" __global__ void v_maxpool_back(const float* g, const long long* arg, float* dx, long long planes,
                                          long long H, long long W, int K, int s, long long Ho, long long Wo) {
    long long i = gid(); if (i >= planes * H * W) return;
    const long long xx = i % W, y = (i / W) % H, pl = i / (W * H);
    const long long oy = y / s, ox = xx / s;
    float r = 0.0f;
    if (oy < Ho && ox < Wo && y - oy * s < K && xx - ox * s < K) {
        const long long o = (pl * Ho + oy) * Wo + ox;
        if (arg[o] == i) r = g[o];
    }
    dx[i] = r;
}
extern "C" __global__ void v_avgpool(const float* x, float* y, long long planes, long long H, long long W,
                                     int K, int s, long long Ho, long long Wo) {
    long long i = gid(); if (i >= planes * Ho * Wo) return;
    const long long ox = i % Wo, oy = (i / Wo) % Ho, pl = i / (Wo * Ho);
    const float* xp = x + pl * H * W;
    float acc = 0.0f;
    for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) acc += xp[(oy * s + kh) * W + ox * s + kw];
    y[i] = acc / (float)(K * K);
}
extern "C" __global__ void v_avgpool_back(const float* g, float* dx, long long planes, long long H, long long W,
                                          int K, int s, long long Ho, long long Wo) {
    long long i = gid(); if (i >= planes * Ho * Wo) return;
    const long long ox = i % Wo, oy = (i / Wo) % Ho, pl = i / (Wo * Ho);
    const float v = g[i] / (float)(K * K);
    for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++) atomicAdd(&dx[pl * H * W + (oy * s + kh) * W + ox * s + kw], v);
}
)");
    return s;
}

// a kernel from VISION_SRC (named VK: these functions have a kernel size K)
static cuda::Kernel VK(const char* name) { return K(name, VISION_SRC()); }

// x [B, C, H, W] -> col [B, C*K*K, Hc*Wc]
static TensorPtr im2col(const TensorPtr& x, long long C, long long H, long long W, int K, int s, int p,
                        long long Hc, long long Wc) {
    long long B = x->size(0);
    auto col = empty_cuda({B, C * K * K, Hc * Wc});
    const void* px = base(x);
    void* pc = col->storage()->data();
    void* args[] = {&px, &pc, &B, &C, &H, &W, &K, &s, &p, &Hc, &Wc};
    cuda::launch(VK("v_im2col"), col->numel(), args);
    return col;
}

// col [B, C*K*K, Hc*Wc] -> x [B, C, H, W]
static TensorPtr col2im(const TensorPtr& col, long long B, long long C, long long H, long long W, int K, int s, int p,
                        long long Hc, long long Wc) {
    auto x = empty_cuda({B, C, H, W});
    const void* pc = base(col);
    void* px = x->storage()->data();
    void* args[] = {&pc, &px, &B, &C, &H, &W, &K, &s, &p, &Hc, &Wc};
    cuda::launch(VK("v_col2im"), x->numel(), args);
    return x;
}

// x [B, C, H, W] -> col [C*K*K, B*Hc*Wc]
static TensorPtr im2col_b(const TensorPtr& x, long long C, long long H, long long W, int K, int s, int p,
                          long long Hc, long long Wc) {
    long long B = x->size(0);
    auto col = empty_cuda({C * K * K, B * Hc * Wc});
    const void* px = base(x);
    void* pc = col->storage()->data();
    void* args[] = {&px, &pc, &B, &C, &H, &W, &K, &s, &p, &Hc, &Wc};
    cuda::launch(VK("v_im2col_b"), col->numel(), args);
    return col;
}

// col [C*K*K, B*Hc*Wc] -> x [B, C, H, W]
static TensorPtr col2im_b(const TensorPtr& col, long long B, long long C, long long H, long long W, int K, int s,
                          int p, long long Hc, long long Wc) {
    auto x = empty_cuda({B, C, H, W});
    const void* pc = base(col);
    void* px = x->storage()->data();
    void* args[] = {&pc, &px, &B, &C, &H, &W, &K, &s, &p, &Hc, &Wc};
    cuda::launch(VK("v_col2im_b"), x->numel(), args);
    return x;
}

// [C, B*P] -> [B, C, P] (+ bias[c]); [B, C, P] -> [C, B*P]
static TensorPtr from_cb(const TensorPtr& y, const TensorPtr& bias, long long B, long long C, long long P,
                         const Shape& shape) {
    auto out = empty_cuda(shape);
    const void* py = base(y);
    const void* pb = bias ? base(bias) : nullptr;
    void* po = out->storage()->data();
    void* args[] = {&py, &pb, &po, &B, &C, &P};
    cuda::launch(VK("v_from_cb"), B * C * P, args);
    return out;
}
static TensorPtr to_cb(const TensorPtr& x, long long B, long long C, long long P) {
    auto out = empty_cuda({C, B * P});
    const void* px = base(x);
    void* po = out->storage()->data();
    void* args[] = {&px, &po, &B, &C, &P};
    cuda::launch(VK("v_to_cb"), B * C * P, args);
    return out;
}

// y [B, C, P] += bias[c] in place; no bias, nothing to do
static void add_bias_c(const TensorPtr& y, const TensorPtr& bias, long long C, long long P) {
    if (!bias) return;
    long long n = y->numel();
    void* py = y->storage()->data();
    const void* pb = base(bias);
    void* args[] = {&py, &pb, &C, &P, &n};
    cuda::launch(VK("v_add_bias_c"), n, args);
}

static const float* fp(const TensorPtr& t) { return static_cast<const float*>(base(t)); }
static float* wp(const TensorPtr& t) { return static_cast<float*>(t->storage()->data()); }

// db for a [B, C, H, W] gradient: the sum over everything but the channel
static TensorPtr bias_grad(const TensorPtr& G, long long B, long long C, long long P) {
    return ::sum(::sum(reshape(G, {B, C, P}), 2), 0);
}

TensorPtr conv2d(const TensorPtr& x0, const TensorPtr& w0, const TensorPtr& b0, int stride, int pad, Reference reference) {
    auto x = contiguous_on_device(f32(x0, "Conv2D")), w = contiguous_on_device(f32(w0, "Conv2D"));
    auto bias = contiguous_on_device(f32(b0, "Conv2D"));
    const long long B = x->size(0), C = x->size(1), H = x->size(2), W = x->size(3);
    const long long F = w->size(0);
    const int K = (int)w->size(2);
    const long long Ho = (H + 2 * pad - K) / stride + 1, Wo = (W + 2 * pad - K) / stride + 1;
    const long long HW = Ho * Wo, CKK = C * K * K, BHW = B * HW;
    if (cuda::dnn_available()) {
        const cuda::ConvShape cs{(int)B, (int)C, (int)H, (int)W, (int)F, K, stride, pad};
        auto out = empty_cuda({B, F, Ho, Wo});
        cuda::dnn_conv_forward(cs, fp(x), fp(w), wp(out));
        add_bias_c(out, bias, F, HW);
        attach_fused(out, {x0, w0, b0}, "Conv2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
            auto G = contiguous_on_device(f32(G0, "Conv2D"));
            auto dx = empty_cuda({B, C, H, W});
            auto dw = empty_cuda({F, C, (long long)K, (long long)K});
            cuda::dnn_conv_backward_data(cs, fp(w), fp(G), wp(dx));
            cuda::dnn_conv_backward_filter(cs, fp(x), fp(G), wp(dw));
            return {dx, dw, bias_grad(G, B, F, HW)};
        });
        return out;
    }
    // the whole batch as one gemm: W [F, CKK] @ col [CKK, B*HW] -> [F, B*HW],
    // then back to [B, F, Ho, Wo] with the bias added in the same pass
    auto col = im2col_b(x, C, H, W, K, stride, pad, Ho, Wo);
    auto yf = empty_cuda({F, BHW});
    cuda::sgemm(false, false, (int)F, (int)BHW, (int)CKK, 1.0f, static_cast<const float*>(base(w)), (int)CKK,
                static_cast<const float*>(base(col)), (int)BHW, 0.0f, static_cast<float*>(yf->storage()->data()), (int)BHW);
    auto out = from_cb(yf, bias, B, F, HW, {B, F, Ho, Wo});
    attach_fused(out, {x0, w0, b0}, "Conv2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "Conv2D"));
        auto gf = to_cb(G, B, F, HW);                                   // [F, B*HW]
        const float* pg = static_cast<const float*>(base(gf));
        // dW = G' @ col^T: one gemm over every image's positions
        auto dw = empty_cuda({F, C, (long long)K, (long long)K});
        cuda::sgemm(false, true, (int)F, (int)CKK, (int)BHW, 1.0f, pg, (int)BHW, static_cast<const float*>(base(col)),
                    (int)BHW, 0.0f, static_cast<float*>(dw->storage()->data()), (int)CKK);
        // dcol = W^T @ G', then back onto the input
        auto dcol = empty_cuda({CKK, BHW});
        cuda::sgemm(true, false, (int)CKK, (int)BHW, (int)F, 1.0f, static_cast<const float*>(base(w)), (int)CKK, pg,
                    (int)BHW, 0.0f, static_cast<float*>(dcol->storage()->data()), (int)BHW);
        auto dx = col2im_b(dcol, B, C, H, W, K, stride, pad, Ho, Wo);
        auto db = ::sum(gf, 1);
        return {dx, dw, db};
    });
    return out;
}

TensorPtr conv_transpose2d(const TensorPtr& x0, const TensorPtr& w0, const TensorPtr& b0, int stride, int pad,
                           Reference reference) {
    auto x = contiguous_on_device(f32(x0, "ConvTranspose2D")), w = contiguous_on_device(f32(w0, "ConvTranspose2D"));
    auto bias = contiguous_on_device(f32(b0, "ConvTranspose2D"));
    const long long B = x->size(0), Cin = x->size(1), H = x->size(2), W = x->size(3);
    const long long Cout = w->size(1);
    const int K = (int)w->size(2);
    const long long Ho = (H - 1) * stride - 2 * pad + K, Wo = (W - 1) * stride - 2 * pad + K;
    const long long HWi = H * W, CKK = Cout * K * K, BHW = B * HWi;
    if (cuda::dnn_available()) {
        // the transpose of a convolution from [B, Cout, Ho, Wo] to [B, Cin, H, W]
        // with filter [Cin, Cout, K, K]: its forward is that convolution's
        // backward-data, and the other way round
        const cuda::ConvShape cs{(int)B, (int)Cout, (int)Ho, (int)Wo, (int)Cin, K, stride, pad};
        auto out = empty_cuda({B, Cout, Ho, Wo});
        cuda::dnn_conv_backward_data(cs, fp(w), fp(x), wp(out));
        add_bias_c(out, bias, Cout, Ho * Wo);
        attach_fused(out, {x0, w0, b0}, "ConvTranspose2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
            auto G = contiguous_on_device(f32(G0, "ConvTranspose2D"));
            auto dx = empty_cuda({B, Cin, H, W});
            auto dw = empty_cuda({Cin, Cout, (long long)K, (long long)K});
            cuda::dnn_conv_forward(cs, fp(G), fp(w), wp(dx));
            cuda::dnn_conv_backward_filter(cs, fp(G), fp(x), wp(dw));
            return {dx, dw, bias_grad(G, B, Cout, Ho * Wo)};
        });
        return out;
    }
    // col = W^T [CKK, Cin] @ x' [Cin, B*HWi], one gemm, then scattered onto the output
    auto xf = to_cb(x, B, Cin, HWi);
    auto col = empty_cuda({CKK, BHW});
    cuda::sgemm(true, false, (int)CKK, (int)BHW, (int)Cin, 1.0f, static_cast<const float*>(base(w)), (int)CKK,
                static_cast<const float*>(base(xf)), (int)BHW, 0.0f, static_cast<float*>(col->storage()->data()), (int)BHW);
    TensorPtr out;
    {
        NoGrad off;
        out = contiguous(add(col2im_b(col, B, Cout, Ho, Wo, K, stride, pad, H, W), reshape(bias, {1, Cout, 1, 1})));
    }
    attach_fused(out, {x0, w0, b0}, "ConvTranspose2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "ConvTranspose2D"));
        auto gc = im2col_b(G, Cout, Ho, Wo, K, stride, pad, H, W);       // [CKK, B*HWi]
        const float* pgc = static_cast<const float*>(base(gc));
        // dx' = W [Cin, CKK] @ gc -> [Cin, B*HWi]; dW = x' @ gc^T
        auto dxf = empty_cuda({Cin, BHW});
        cuda::sgemm(false, false, (int)Cin, (int)BHW, (int)CKK, 1.0f, static_cast<const float*>(base(w)), (int)CKK, pgc,
                    (int)BHW, 0.0f, static_cast<float*>(dxf->storage()->data()), (int)BHW);
        auto dw = empty_cuda({Cin, Cout, (long long)K, (long long)K});
        cuda::sgemm(false, true, (int)Cin, (int)CKK, (int)BHW, 1.0f, static_cast<const float*>(base(xf)), (int)BHW, pgc,
                    (int)BHW, 0.0f, static_cast<float*>(dw->storage()->data()), (int)CKK);
        auto dx = from_cb(dxf, nullptr, B, Cin, HWi, {B, Cin, H, W});
        auto db = ::sum(::sum(reshape(G, {B, Cout, Ho * Wo}), 2), 0);
        return {dx, dw, db};
    });
    return out;
}

TensorPtr maxpool2d(const TensorPtr& x0, int K, int stride) {
    auto x = contiguous_on_device(f32(x0, "MaxPool2D"));
    long long Bn = x->size(0), C = x->size(1), H = x->size(2), W = x->size(3);
    long long Ho = (H - K) / stride + 1, Wo = (W - K) / stride + 1, planes = Bn * C;
    auto y = empty_cuda({Bn, C, Ho, Wo});
    auto arg = empty_cuda({Bn * C * Ho * Wo}, DType::Int64);
    const void* px = base(x);
    void* py = y->storage()->data();
    void* pa = arg->storage()->data();
    int k = K, s = stride;
    void* args[] = {&px, &py, &pa, &planes, &H, &W, &k, &s, &Ho, &Wo};
    cuda::launch(VK("v_maxpool"), y->numel(), args);
    const Shape oshape{Bn, C, Ho, Wo};
    // op form: a gather of the winners (indices read back only for it)
    Reference reference = [arg, oshape](const vector<TensorPtr>& in) -> TensorPtr {
        auto h = arg->to(Device::CPU);
        auto idx = make_shared<vector<int64_t>>(h->numel());
        memcpy(idx->data(), h->storage()->data(), idx->size() * sizeof(int64_t));
        return gather_flat(in[0], idx, oshape);
    };
    const int64_t nin = x->numel();
    const Shape xshape = x->shape();
    attach_fused(y, {x0}, "MaxPool2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "MaxPool2D"));
        if (stride >= K) {
            auto dx = empty_cuda(xshape);
            const void* pg = base(G);
            const void* pa = arg->storage()->data();
            void* pd = dx->storage()->data();
            long long pls = planes, h = H, w = W, ho = Ho, wo = Wo;
            int k2 = K, s2 = stride;
            void* a[] = {&pg, &pa, &pd, &pls, &h, &w, &k2, &s2, &ho, &wo};
            cuda::launch(VK("v_maxpool_back"), dx->numel(), a);
            return {dx};
        }
        auto dx = make_shared<Tensor>(xshape, F32, Device::CUDA);        // zeroed
        const void* pg = base(G);
        const void* pa = arg->storage()->data();
        void* pd = dx->storage()->data();
        long long n = G->numel();
        void* a[] = {&pg, &pa, &pd, &n};
        cuda::launch(VK("v_scatter_rows"), n, a);
        (void)nin;
        return {dx};
    });
    return y;
}

TensorPtr avgpool2d(const TensorPtr& x0, int K, int stride, Reference reference) {
    auto x = contiguous_on_device(f32(x0, "AvgPool2D"));
    long long Bn = x->size(0), C = x->size(1), H = x->size(2), W = x->size(3);
    long long Ho = (H - K) / stride + 1, Wo = (W - K) / stride + 1, planes = Bn * C;
    auto y = empty_cuda({Bn, C, Ho, Wo});
    const void* px = base(x);
    void* py = y->storage()->data();
    int k = K, s = stride;
    void* args[] = {&px, &py, &planes, &H, &W, &k, &s, &Ho, &Wo};
    cuda::launch(VK("v_avgpool"), y->numel(), args);
    const Shape xshape = x->shape();
    attach_fused(y, {x0}, "AvgPool2D", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "AvgPool2D"));
        auto dx = make_shared<Tensor>(xshape, F32, Device::CUDA);        // zeroed
        const void* pg = base(G);
        void* pd = dx->storage()->data();
        long long pl = planes, hh = H, ww = W, ho = Ho, wo = Wo;
        int kk = K, ss = stride;
        void* a[] = {&pg, &pd, &pl, &hh, &ww, &kk, &ss, &ho, &wo};
        cuda::launch(VK("v_avgpool_back"), G->numel(), a);
        return {dx};
    });
    return y;
}

// ── BatchNorm and GroupNorm on the device ────────────────────────────────────
//
// BatchNorm: per channel, one block reduces the N*S values (mean, then the
// variance about it) and updates the running estimate in device memory, so a
// training step never waits to read statistics back; a second kernel applies
// the normalisation. The backward is the same shape: one block per channel
// for sum(dy) and sum(dy * xhat), then an element-wise kernel for dx.
// GroupNorm: an (image, group) is one contiguous run of group_size * S values,
// so it is a LayerNorm over that run with a per-channel affine.

static const string& NORM_SRC() {
    static const string s = src(R"(
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w]; __syncthreads(); }
    const float r = sh[0];
    __syncthreads();
    return r;
}
// one block per channel of x viewed as [N, C, S]: mean and 1/std, and the running estimate
extern "C" __global__ void n_bn_stats(const float* x, float* mean, float* inv, float* rm, float* rv,
                                      long long N, long long C, long long S, float eps, float momentum, int update) {
    __shared__ float sh[256];
    const long long c = blockIdx.x, count = N * S;
    float t = 0.0f;
    for (long long i = threadIdx.x; i < count; i += blockDim.x) t += x[((i / S) * C + c) * S + i % S];
    const float m = block_sum(t, sh) / (float)count;
    t = 0.0f;
    for (long long i = threadIdx.x; i < count; i += blockDim.x) { float d = x[((i / S) * C + c) * S + i % S] - m; t += d * d; }
    const float var = block_sum(t, sh) / (float)count;
    if (threadIdx.x == 0) {
        mean[c] = m;
        inv[c] = rsqrtf(var + eps);
        if (update) {
            const float unbiased = count > 1 ? var * (float)count / (float)(count - 1) : var;
            rm[c] = (1.0f - momentum) * rm[c] + momentum * m;
            rv[c] = (1.0f - momentum) * rv[c] + momentum * unbiased;
        }
    }
}
// eval: the statistics are the running estimate
extern "C" __global__ void n_bn_running(const float* rm, const float* rv, float* mean, float* inv, long long C, float eps) {
    long long c = gid(); if (c >= C) return;
    mean[c] = rm[c];
    inv[c] = rsqrtf(rv[c] + eps);
}
extern "C" __global__ void n_affine(const float* x, const float* mean, const float* inv, const float* g, const float* b,
                                    float* y, long long C, long long S, long long n) {
    long long i = gid(); if (i >= n) return;
    const long long c = (i / S) % C;
    y[i] = (x[i] - mean[c]) * inv[c] * g[c] + b[c];
}
// backward sums per channel: sum(dy), sum(dy * xhat) -> also beta's and gamma's gradients
extern "C" __global__ void n_bn_back_sums(const float* x, const float* dy, const float* mean, const float* inv,
                                          float* sdy, float* sdyx, long long N, long long C, long long S) {
    __shared__ float sh[256];
    const long long c = blockIdx.x, count = N * S;
    float a = 0.0f, b = 0.0f;
    for (long long i = threadIdx.x; i < count; i += blockDim.x) {
        const long long j = ((i / S) * C + c) * S + i % S;
        a += dy[j];
        b += dy[j] * (x[j] - mean[c]) * inv[c];
    }
    a = block_sum(a, sh);
    b = block_sum(b, sh);
    if (threadIdx.x == 0) { sdy[c] = a; sdyx[c] = b; }
}
// dx = inv * g * (dy - mean(dy) - xhat * mean(dy * xhat)) in training; inv * g * dy in eval
extern "C" __global__ void n_bn_back_dx(const float* x, const float* dy, const float* mean, const float* inv,
                                        const float* g, const float* sdy, const float* sdyx, float* dx,
                                        long long C, long long S, long long count, long long n, int training) {
    long long i = gid(); if (i >= n) return;
    const long long c = (i / S) % C;
    const float k = inv[c] * g[c];
    if (!training) { dx[i] = k * dy[i]; return; }
    const float xh = (x[i] - mean[c]) * inv[c];
    dx[i] = k * (dy[i] - sdy[c] / (float)count - xh * sdyx[c] / (float)count);
}
// GroupNorm forward: one block per (image, group), a contiguous run of L = gs * S values
extern "C" __global__ void n_gn_forward(const float* x, const float* g, const float* b, float* y, float* mean, float* inv,
                                        long long gs, long long S, long long C, float eps) {
    __shared__ float sh[256];
    const long long L = gs * S, off = (long long)blockIdx.x * L;
    const long long c0 = (blockIdx.x * gs) % C;
    float t = 0.0f;
    for (long long j = threadIdx.x; j < L; j += blockDim.x) t += x[off + j];
    const float m = block_sum(t, sh) / (float)L;
    t = 0.0f;
    for (long long j = threadIdx.x; j < L; j += blockDim.x) { float d = x[off + j] - m; t += d * d; }
    const float iv = rsqrtf(block_sum(t, sh) / (float)L + eps);
    for (long long j = threadIdx.x; j < L; j += blockDim.x) {
        const long long c = c0 + j / S;
        y[off + j] = (x[off + j] - m) * iv * g[c] + b[c];
    }
    if (threadIdx.x == 0) { mean[blockIdx.x] = m; inv[blockIdx.x] = iv; }
}
// GroupNorm backward dx: per (image, group), dx = inv * (g*dy - mean(g*dy) - xhat * mean(g*dy*xhat))
extern "C" __global__ void n_gn_back(const float* x, const float* g, const float* dy, const float* mean, const float* inv,
                                     float* dx, long long gs, long long S, long long C) {
    __shared__ float sh[256];
    const long long L = gs * S, off = (long long)blockIdx.x * L;
    const long long c0 = (blockIdx.x * gs) % C;
    const float m = mean[blockIdx.x], iv = inv[blockIdx.x];
    float a = 0.0f, b = 0.0f;
    for (long long j = threadIdx.x; j < L; j += blockDim.x) {
        const float gd = g[c0 + j / S] * dy[off + j];
        a += gd;
        b += gd * (x[off + j] - m) * iv;
    }
    a = block_sum(a, sh) / (float)L;
    b = block_sum(b, sh) / (float)L;
    for (long long j = threadIdx.x; j < L; j += blockDim.x) {
        const float xh = (x[off + j] - m) * iv;
        dx[off + j] = iv * (g[c0 + j / S] * dy[off + j] - a - xh * b);
    }
}
// GroupNorm gamma / beta: a thread per (image, channel), summed over S, added atomically
extern "C" __global__ void n_gn_back_affine(const float* x, const float* dy, const float* mean, const float* inv,
                                            float* dg, float* db, long long B, long long C, long long gs, long long S) {
    long long i = gid(); if (i >= B * C) return;
    const long long c = i % C, bimg = i / C, grp = bimg * (C / gs) + c / gs;
    const float m = mean[grp], iv = inv[grp];
    const float* xr = x + i * S;
    const float* dr = dy + i * S;
    float ag = 0.0f, ab = 0.0f;
    for (long long s = 0; s < S; s++) { ag += dr[s] * (xr[s] - m) * iv; ab += dr[s]; }
    atomicAdd(&dg[c], ag);
    atomicAdd(&db[c], ab);
}
)");
    return s;
}

static cuda::Kernel NK(const char* name) { return K(name, NORM_SRC()); }

TensorPtr batchnorm(const TensorPtr& x0, const TensorPtr& g0, const TensorPtr& b0, const TensorPtr& rm,
                    const TensorPtr& rv, bool training, float momentum, float eps, Reference reference) {
    auto x = contiguous_on_device(f32(x0, "BatchNorm"));
    auto g = contiguous_on_device(f32(g0, "BatchNorm")), b = contiguous_on_device(f32(b0, "BatchNorm"));
    long long N = x->size(0), C = x->size(1), S = x->numel() / max<long long>(N * C, 1), n = x->numel();
    auto mean = empty_cuda({C}), inv = empty_cuda({C}), y = empty_cuda(x->shape());
    const void* px = base(x);
    void* pm = mean->storage()->data();
    void* pi = inv->storage()->data();
    void* prm = rm->storage()->data();
    void* prv = rv->storage()->data();
    if (training) {
        int upd = 1;
        float mo = momentum, ep = eps;
        void* a[] = {&px, &pm, &pi, &prm, &prv, &N, &C, &S, &ep, &mo, &upd};
        cuda::launch(NK("n_bn_stats"), (unsigned)C, 1, 1, 256, 1, 1, 0, a);
    } else {
        float ep = eps;
        void* a[] = {&prm, &prv, &pm, &pi, &C, &ep};
        cuda::launch(NK("n_bn_running"), C, a);
    }
    const void* pg = base(g);
    const void* pb = base(b);
    void* py = y->storage()->data();
    void* a2[] = {&px, &pm, &pi, &pg, &pb, &py, &C, &S, &n};
    cuda::launch(NK("n_affine"), n, a2);
    attach_fused(y, {x0, g0, b0}, "BatchNorm", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "BatchNorm"));
        auto sdy = empty_cuda({C}), sdyx = empty_cuda({C}), dx = empty_cuda(x->shape());
        const void* px2 = base(x);
        const void* pg2 = base(G);
        const void* pm2 = base(mean);
        const void* pi2 = base(inv);
        void* ps1 = sdy->storage()->data();
        void* ps2 = sdyx->storage()->data();
        long long NN = N, CC = C, SS = S, nn = n, count = N * S;
        void* a3[] = {&px2, &pg2, &pm2, &pi2, &ps1, &ps2, &NN, &CC, &SS};
        cuda::launch(NK("n_bn_back_sums"), (unsigned)C, 1, 1, 256, 1, 1, 0, a3);
        const void* pgam = base(g);
        void* pdx = dx->storage()->data();
        int tr = training ? 1 : 0;
        void* a4[] = {&px2, &pg2, &pm2, &pi2, &pgam, &ps1, &ps2, &pdx, &CC, &SS, &count, &nn, &tr};
        cuda::launch(NK("n_bn_back_dx"), nn, a4);
        return {dx, sdyx, sdy};          // gamma's gradient is sum(dy*xhat), beta's sum(dy)
    });
    return y;
}

TensorPtr groupnorm(const TensorPtr& x0, const TensorPtr& g0, const TensorPtr& b0, int groups, float eps,
                    Reference reference) {
    auto x = contiguous_on_device(f32(x0, "GroupNorm"));
    auto g = contiguous_on_device(f32(g0, "GroupNorm")), b = contiguous_on_device(f32(b0, "GroupNorm"));
    long long B = x->size(0), C = x->size(1), S = x->numel() / max<long long>(B * C, 1), gs = C / groups;
    const long long blocks = B * groups;
    auto y = empty_cuda(x->shape()), mean = empty_cuda({blocks}), inv = empty_cuda({blocks});
    const void* px = base(x);
    const void* pg = base(g);
    const void* pb = base(b);
    void* py = y->storage()->data();
    void* pm = mean->storage()->data();
    void* pi = inv->storage()->data();
    float ep = eps;
    void* a[] = {&px, &pg, &pb, &py, &pm, &pi, &gs, &S, &C, &ep};
    if (blocks) cuda::launch(NK("n_gn_forward"), (unsigned)blocks, 1, 1, 256, 1, 1, 0, a);
    attach_fused(y, {x0, g0, b0}, "GroupNorm", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "GroupNorm"));
        auto dx = empty_cuda(x->shape());
        auto dg = make_shared<Tensor>(Shape{C}, F32, Device::CUDA), db = make_shared<Tensor>(Shape{C}, F32, Device::CUDA);
        const void* px2 = base(x);
        const void* pg2 = base(g);
        const void* pd = base(G);
        const void* pm2 = base(mean);
        const void* pi2 = base(inv);
        void* pdx = dx->storage()->data();
        long long gss = gs, SS = S, CC = C, BB = B;
        void* a1[] = {&px2, &pg2, &pd, &pm2, &pi2, &pdx, &gss, &SS, &CC};
        if (blocks) cuda::launch(NK("n_gn_back"), (unsigned)blocks, 1, 1, 256, 1, 1, 0, a1);
        void* pdg = dg->storage()->data();
        void* pdb = db->storage()->data();
        void* a2[] = {&px2, &pd, &pm2, &pi2, &pdg, &pdb, &BB, &CC, &gss, &SS};
        cuda::launch(NK("n_gn_back_affine"), B * C, a2);
        return {dx, dg, db};
    });
    return y;
}

} // namespace cuda_ops
