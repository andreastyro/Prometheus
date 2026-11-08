#include "ml/nn/rnn/fused_rnn.hpp"
#include "ml/matmul_backend.hpp"
#include "ml/ops.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

using namespace std;

namespace rnn_kernel {
namespace {

inline float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// C = A @ op(B) + beta*C, split across this library's own threads, each slice
// a single-threaded BLAS call.
//
// The recurrence alternates small gemms with elementwise gate loops 64 times a
// sequence. Handing the gemms to a threaded BLAS meant two thread pools --
// MKL's and ours -- taking turns, each spin-waiting through the other's work;
// it made the per-step gemm 1.7x slower than when the gate loops were serial.
// With every gemm on our own threads, BLAS's pool never wakes in here.
void pgemm(bool ta, bool tb, int M, int N, int K, const float* A, int lda,
           const float* Bm, int ldb, float beta, float* C, int ldc) {
    const int T = matmul_max_threads();
    if (T <= 1 || (int64_t)M * N * K < (1 << 18)) {
        gemm_serial(ta, tb, M, N, K, A, lda, Bm, ldb, beta, C, ldc);
        return;
    }
    // split the larger output dimension, so each thread's tile stays close to
    // square. Splitting rows whenever M >= 4T gave the LSTM weight gradient
    // [128, 1024] tiles of 8 x 1024 with K = 2048: 7.6 ms where 128 x 64
    // tiles take a third of that.
    const bool rows = M >= N;
    const int D = rows ? M : N;
    const int parts = std::min(T, std::max(1, D / 8));
    SerialBlasScope serial;
    #pragma omp parallel for schedule(static)
    for (int q = 0; q < parts; q++) {
        const int lo = (int)((int64_t)D * q / parts), hi = (int)((int64_t)D * (q + 1) / parts);
        if (hi <= lo) continue;
        if (rows)
            gemm_serial(ta, tb, hi - lo, N, K, ta ? A + lo : A + (int64_t)lo * lda, lda,
                        Bm, ldb, beta, C + (int64_t)lo * ldc, ldc);
        else
            gemm_serial(ta, tb, M, hi - lo, K, A, lda, tb ? Bm + (int64_t)lo * ldb : Bm + lo, ldb,
                        beta, C + lo, ldc);
    }
}

// A contiguous slice of `src`'s storage as its own tensor, differentiable: its
// gradient is added back into src's at the same offset. The layer's outputs --
// the sequence, its last step and the final cell state -- are slices of one
// packed buffer owned by the kernel's node.
TensorPtr slice_of(const TensorPtr& src, int64_t offset, Shape shape) {
    auto v = make_shared<Tensor>(src->storage(), shape, Tensor::contiguous_strides(shape),
                                 src->offset() + offset);
    if (src->requires_grad()) {
        auto node = make_node(v, {src});
        node->name = "rnn output";
        autograd::attach_reference(*node, [offset, shape](const std::vector<TensorPtr>& in) -> TensorPtr {
            auto idx = make_shared<vector<int64_t>>();
            int64_t n = 1; for (auto v : shape) n *= v;
            for (int64_t i = 0; i < n; i++) idx->push_back(offset + i);
            return gather_flat(in[0], idx, shape);
        });
        node->backward_fn = [src, offset](const TensorPtr& out) {
            const float* g = out->grad();
            float* gs = src->grad() + offset;
            for (int64_t i = 0; i < out->numel(); i++) gs[i] += g[i];
        };
    }
    return v;
}

// [rows, H] per gate, side by side -> [rows, G*H]
vector<float> pack_cols(const vector<TensorPtr>& parts, int rows, int H) {
    const int G = (int)parts.size();
    vector<float> out((size_t)rows * G * H);
    for (int k = 0; k < G; k++) {
        const float* src = parts[k]->data();
        for (int r = 0; r < rows; r++)
            memcpy(&out[((size_t)r * G + k) * H], src + (size_t)r * H, H * sizeof(float));
    }
    return out;
}

// the reverse: add a packed [rows, G*H] gradient into each gate's [rows, H]
void unpack_add(const float* packed, const vector<TensorPtr>& parts, int rows, int H, int G0, int Gtot) {
    for (int k = 0; k < (int)parts.size(); k++) {
        if (!parts[k]->requires_grad()) continue;
        float* dst = parts[k]->grad();
        for (int r = 0; r < rows; r++) {
            const float* src = packed + (size_t)r * Gtot * H + (size_t)(G0 + k) * H;
            for (int j = 0; j < H; j++) dst[(size_t)r * H + j] += src[j];
        }
    }
}

} // namespace

TensorPtr init_param(Shape shape, int H) {
    auto t = make_shared<Tensor>(shape);
    const float bound = 1.0f / std::sqrt((float)H);
    std::uniform_real_distribution<float> dist(-bound, bound);
    for (float& v : t->values()) v = dist(prometheus_rng());
    t->set_requires_grad(true);
    return t;
}

Result run(Cell cell, TensorPtr input, TensorPtr h0, TensorPtr c0, int H,
           const vector<TensorPtr>& Wx, const vector<TensorPtr>& Wh, const vector<TensorPtr>& b) {
    const char* name = cell == Cell::RNN ? "RNN" : cell == Cell::LSTM ? "LSTM" : "GRU";
    input = checked_input(name, input);
    if (input->dim() != 3)
        throw runtime_error(string(name) + ": input must be [seq, batch, features]");
    const int S = (int)input->size(0), B = (int)input->size(1), I = (int)input->size(2);
    if (I != (int)Wx[0]->size(0))
        throw runtime_error(string(name) + ": expected " + to_string(Wx[0]->size(0)) +
                            " input features, got " + to_string(I));
    const int G = (int)Wx.size(), GH = G * H, BH = B * H;
    const int64_t SB = (int64_t)S * B;
    if (h0) h0 = checked_input(name, h0);
    if (c0) c0 = checked_input(name, c0);

    // packed weights: gates side by side
    auto Wxp = make_shared<vector<float>>(pack_cols(Wx, I, H));         // [I, GH]
    auto Whp = make_shared<vector<float>>(pack_cols(Wh, H, H));         // [H, GH]
    vector<float> bp(GH);
    for (int k = 0; k < G; k++) memcpy(&bp[(size_t)k * H], b[k]->data(), H * sizeof(float));

    // 1. every timestep's input projection, plus bias, in one gemm
    auto XP = make_shared<vector<float>>((size_t)SB * GH);
    for (int64_t r = 0; r < SB; r++) memcpy(&(*XP)[(size_t)r * GH], bp.data(), GH * sizeof(float));
    pgemm(false, false, (int)SB, GH, I, input->data(), I, Wxp->data(), GH, 1.0f, XP->data(), GH);

    // the outputs live in one packed buffer: h for every step, then c_n
    const bool lstm = cell == Cell::LSTM;
    auto packed = make_shared<Tensor>(Shape{SB * H + (lstm ? BH : 0)});
    float* OUT = packed->data();

    // saved for the backward pass
    auto gates = make_shared<vector<float>>((size_t)SB * GH);   // activated gates per step
    auto cell_state = lstm ? make_shared<vector<float>>((size_t)SB * H) : nullptr;
    auto rh = cell == Cell::GRU ? make_shared<vector<float>>((size_t)SB * H) : nullptr;

    vector<float> zero_h((size_t)BH, 0.0f);
    const float* h_init = h0 ? h0->data() : zero_h.data();
    const float* c_init = (lstm && c0) ? c0->data() : zero_h.data();
    vector<float> a((size_t)B * GH);

    // 2. the recurrence: one gemm per step, then the gate arithmetic fused
    for (int t = 0; t < S; t++) {
        const float* h_prev = t ? OUT + (size_t)(t - 1) * BH : h_init;
        const float* xp = XP->data() + (size_t)t * B * GH;
        float* gt = gates->data() + (size_t)t * B * GH;
        float* ht = OUT + (size_t)t * BH;
        memcpy(a.data(), xp, a.size() * sizeof(float));

        if (cell == Cell::RNN) {
            pgemm(false, false, B, H, H, h_prev, H, Whp->data(), GH, 1.0f, a.data(), GH);
            #pragma omp parallel for schedule(static) if(BH >= 4096)
            for (int i = 0; i < BH; i++) ht[i] = gt[i] = std::tanh(a[i]);
        } else if (lstm) {
            pgemm(false, false, B, GH, H, h_prev, H, Whp->data(), GH, 1.0f, a.data(), GH);
            const float* c_prev = t ? cell_state->data() + (size_t)(t - 1) * BH : c_init;
            float* ct = cell_state->data() + (size_t)t * BH;
            #pragma omp parallel for schedule(static) if(BH >= 4096)
            for (int bb = 0; bb < B; bb++) {
                const float* ar = a.data() + (size_t)bb * GH;
                float* gr = gt + (size_t)bb * GH;
                for (int j = 0; j < H; j++) {
                    const float i_ = sigm(ar[j]), f_ = sigm(ar[H + j]);
                    const float o_ = sigm(ar[2 * H + j]), g_ = std::tanh(ar[3 * H + j]);
                    gr[j] = i_; gr[H + j] = f_; gr[2 * H + j] = o_; gr[3 * H + j] = g_;
                    const int64_t k = (int64_t)bb * H + j;
                    ct[k] = f_ * c_prev[k] + i_ * g_;
                    ht[k] = o_ * std::tanh(ct[k]);
                }
            }
        } else {
            // r and z from h; n from (r * h)
            const int H2 = 2 * H;
            pgemm(false, false, B, H2, H, h_prev, H, Whp->data(), GH, 1.0f, a.data(), GH);
            float* rht = rh->data() + (size_t)t * BH;
            #pragma omp parallel for schedule(static) if(BH >= 4096)
            for (int bb = 0; bb < B; bb++) {
                const float* ar = a.data() + (size_t)bb * GH;
                float* gr = gt + (size_t)bb * GH;
                for (int j = 0; j < H; j++) {
                    gr[j] = sigm(ar[j]);
                    gr[H + j] = sigm(ar[H + j]);
                    rht[(size_t)bb * H + j] = gr[j] * h_prev[(size_t)bb * H + j];
                }
            }
            pgemm(false, false, B, H, H, rht, H, Whp->data() + H2, GH, 1.0f, a.data() + H2, GH);
            #pragma omp parallel for schedule(static) if(BH >= 4096)
            for (int bb = 0; bb < B; bb++) {
                const float* ar = a.data() + (size_t)bb * GH;
                float* gr = gt + (size_t)bb * GH;
                for (int j = 0; j < H; j++) {
                    const float n_ = std::tanh(ar[H2 + j]), z_ = gr[H + j];
                    gr[H2 + j] = n_;
                    const int64_t k = (int64_t)bb * H + j;
                    ht[k] = (1.0f - z_) * n_ + z_ * h_prev[k];
                }
            }
        }
    }
    if (lstm) memcpy(OUT + SB * H, cell_state->data() + (size_t)(S - 1) * BH, BH * sizeof(float));

    vector<TensorPtr> inputs = {input};
    if (h0) inputs.push_back(h0);
    if (c0) inputs.push_back(c0);
    for (auto* group : {&Wx, &Wh, &b}) for (auto& w : *group) inputs.push_back(w);
    bool need = false;
    for (auto& t : inputs) need = need || t->requires_grad();

    if (need) {
        auto node = make_node(packed, inputs);
        node->name = name;
        autograd::attach_reference(*node, [cell, S, B, I, H, G, lstm, has_h0 = (bool)h0, has_c0 = (bool)c0](const std::vector<TensorPtr>& in) -> TensorPtr {
            // the recurrence unrolled from ops, in the cell's own formulas
            size_t k = 0;
            auto X = in[k++];
            TensorPtr h = has_h0 ? in[k++] : Tensor::zeros({(int64_t)B, (int64_t)H});
            TensorPtr c = (lstm && has_c0) ? in[k++] : Tensor::zeros({(int64_t)B, (int64_t)H});
            vector<TensorPtr> Wx(in.begin() + k, in.begin() + k + G); k += G;
            vector<TensorPtr> Wh(in.begin() + k, in.begin() + k + G); k += G;
            vector<TensorPtr> bb(in.begin() + k, in.begin() + k + G);
            auto lin = [&](const TensorPtr& x, const TensorPtr& hh, int g) {
                return add(add(matmul(x, Wx[g]), matmul(hh, Wh[g])), bb[g]);
            };
            vector<TensorPtr> hs;
            for (int t = 0; t < S; t++) {
                auto idx = make_shared<vector<int64_t>>((size_t)B * I);
                for (int64_t q = 0; q < (int64_t)B * I; q++) (*idx)[q] = (int64_t)t * B * I + q;
                auto x = gather_flat(X, idx, {(int64_t)B, (int64_t)I});
                if (cell == Cell::RNN) {
                    h = tanh_op(lin(x, h, 0));
                } else if (lstm) {
                    auto ig = sigmoid(lin(x, h, 0)), fg = sigmoid(lin(x, h, 1));
                    auto og = sigmoid(lin(x, h, 2)), gg = tanh_op(lin(x, h, 3));
                    c = add(multiply(fg, c), multiply(ig, gg));
                    h = multiply(og, tanh_op(c));
                } else {
                    auto r = sigmoid(lin(x, h, 0)), z = sigmoid(lin(x, h, 1));
                    auto n = tanh_op(add(add(matmul(x, Wx[2]), matmul(multiply(r, h), Wh[2])), bb[2]));
                    h = add(multiply(subtract(1.0f, z), n), multiply(z, h));
                }
                hs.push_back(reshape(h, {(int64_t)B * H}));
            }
            auto seq = concat_op(hs, 0);
            return lstm ? concat_op({seq, reshape(c, {(int64_t)B * H})}, 0) : seq;
        });
        node->backward_fn = [cell, input, h0, c0, H, S, B, I, G, GH, BH, SB, lstm,
                             Wxp, Whp, gates, cell_state, rh, Wx, Wh, b]
                            (const TensorPtr& packed) {
            const float* OUTd = packed->data();
            const float* dOUT = packed->grad();
            vector<float> zero((size_t)BH, 0.0f);
            const float* h_init = h0 ? h0->data() : zero.data();
            const float* c_init = (lstm && c0) ? c0->data() : zero.data();

            // dA: gradient of each step's pre-activations, [SB, GH]
            vector<float> dA((size_t)SB * GH);
            vector<float> dh((size_t)BH, 0.0f), dc((size_t)BH, 0.0f), drh((size_t)BH);
            if (lstm) memcpy(dc.data(), dOUT + SB * H, BH * sizeof(float));

            // 3. backward through time
            for (int t = S - 1; t >= 0; t--) {
                const float* h_prev = t ? OUTd + (size_t)(t - 1) * BH : h_init;
                const float* gt = gates->data() + (size_t)t * B * GH;
                float* da = dA.data() + (size_t)t * B * GH;
                for (int i = 0; i < BH; i++) dh[i] += dOUT[(size_t)t * BH + i];

                if (cell == Cell::RNN) {
                    #pragma omp parallel for schedule(static) if(BH >= 4096)
                    for (int i = 0; i < BH; i++) da[i] = dh[i] * (1.0f - gt[i] * gt[i]);
                    fill(dh.begin(), dh.end(), 0.0f);
                    pgemm(false, true, B, H, GH, da, GH, Whp->data(), GH, 1.0f, dh.data(), H);
                } else if (lstm) {
                    const float* ct = cell_state->data() + (size_t)t * BH;
                    const float* c_prev = t ? cell_state->data() + (size_t)(t - 1) * BH : c_init;
            #pragma omp parallel for schedule(static) if(BH >= 4096)
            for (int bb = 0; bb < B; bb++) {
                        const float* gr = gt + (size_t)bb * GH;
                        float* dr = da + (size_t)bb * GH;
                        for (int j = 0; j < H; j++) {
                            const int64_t k = (int64_t)bb * H + j;
                            const float i_ = gr[j], f_ = gr[H + j], o_ = gr[2 * H + j], g_ = gr[3 * H + j];
                            const float tc = std::tanh(ct[k]);
                            const float dct = dc[k] + dh[k] * o_ * (1.0f - tc * tc);
                            dr[j]         = dct * g_ * i_ * (1.0f - i_);
                            dr[H + j]     = dct * c_prev[k] * f_ * (1.0f - f_);
                            dr[2 * H + j] = dh[k] * tc * o_ * (1.0f - o_);
                            dr[3 * H + j] = dct * i_ * (1.0f - g_ * g_);
                            dc[k] = dct * f_;
                        }
                    }
                    fill(dh.begin(), dh.end(), 0.0f);
                    pgemm(false, true, B, H, GH, da, GH, Whp->data(), GH, 1.0f, dh.data(), H);
                } else {
                    const int H2 = 2 * H;
                    vector<float> dh_prev((size_t)BH);
            #pragma omp parallel for schedule(static) if(BH >= 4096)
            for (int bb = 0; bb < B; bb++) {
                        const float* gr = gt + (size_t)bb * GH;
                        float* dr = da + (size_t)bb * GH;
                        for (int j = 0; j < H; j++) {
                            const int64_t k = (int64_t)bb * H + j;
                            const float z_ = gr[H + j], n_ = gr[H2 + j];
                            dr[H2 + j] = dh[k] * (1.0f - z_) * (1.0f - n_ * n_);   // n pre-activation
                            dr[H + j]  = dh[k] * (h_prev[k] - n_) * z_ * (1.0f - z_);
                            dh_prev[k] = dh[k] * z_;
                        }
                    }
                    // through (r * h) @ W_hn
                    pgemm(false, true, B, H, H, da + H2, GH, Whp->data() + H2, GH, 0.0f, drh.data(), H);
            #pragma omp parallel for schedule(static) if(BH >= 4096)
            for (int bb = 0; bb < B; bb++) {
                        const float* gr = gt + (size_t)bb * GH;
                        float* dr = da + (size_t)bb * GH;
                        for (int j = 0; j < H; j++) {
                            const int64_t k = (int64_t)bb * H + j;
                            const float r_ = gr[j];
                            dr[j] = drh[k] * h_prev[k] * r_ * (1.0f - r_);
                            dh_prev[k] += drh[k] * r_;
                        }
                    }
                    // through h @ [W_hr W_hz]
                    pgemm(false, true, B, H, H2, da, GH, Whp->data(), GH, 1.0f, dh_prev.data(), H);
                    dh.swap(dh_prev);
                }
            }

            // 4. everything over all timesteps at once
            if (input->requires_grad())       // dX [SB, I] += dA @ Wx^T
                pgemm(false, true, (int)SB, I, GH, dA.data(), GH, Wxp->data(), GH, 1.0f, input->grad(), I);
            if (h0 && h0->requires_grad())
                for (int i = 0; i < BH; i++) h0->grad()[i] += dh[i];
            if (lstm && c0 && c0->requires_grad())
                for (int i = 0; i < BH; i++) c0->grad()[i] += dc[i];

            bool wx = false, wh = false, wb = false;
            for (auto& w : Wx) wx = wx || w->requires_grad();
            for (auto& w : Wh) wh = wh || w->requires_grad();
            for (auto& w : b)  wb = wb || w->requires_grad();

            if (wx) {                         // dWx [I, GH] = X^T @ dA
                vector<float> d((size_t)I * GH, 0.0f);
                pgemm(true, false, I, GH, (int)SB, input->data(), I, dA.data(), GH, 0.0f, d.data(), GH);
                unpack_add(d.data(), Wx, I, H, 0, G);
            }
            if (wb) {
                // column sums of dA [SB, GH]: each thread a band of columns,
                // rows summed in float blocks of 64 then folded into double.
                // It was one serial pass: 3.3 ms of a 24 ms LSTM step.
                vector<double> d((size_t)GH, 0.0);
                constexpr int CB = 64;
                #pragma omp parallel for schedule(static) if(SB * GH >= 65536)
                for (int j0 = 0; j0 < GH; j0 += CB) {
                    const int j1 = std::min(GH, j0 + CB);
                    float acc[CB];
                    for (int64_t r0 = 0; r0 < SB; r0 += 64) {
                        const int64_t r1 = std::min<int64_t>(SB, r0 + 64);
                        for (int j = j0; j < j1; j++) acc[j - j0] = 0.0f;
                        for (int64_t r = r0; r < r1; r++) {
                            const float* row = dA.data() + (size_t)r * GH;
                            for (int j = j0; j < j1; j++) acc[j - j0] += row[j];
                        }
                        for (int j = j0; j < j1; j++) d[j] += acc[j - j0];
                    }
                }
                for (int k = 0; k < G; k++)
                    if (b[k]->requires_grad())
                        for (int j = 0; j < H; j++) b[k]->grad()[j] += (float)d[(size_t)k * H + j];
            }
            if (wh) {
                // h_{t-1} for every step, [SB, H]: h0 then the outputs shifted by one
                vector<float> Hprev((size_t)SB * H);
                memcpy(Hprev.data(), h0 ? h0->data() : zero.data(), BH * sizeof(float));
                if (S > 1) memcpy(Hprev.data() + BH, OUTd, (size_t)(S - 1) * BH * sizeof(float));
                vector<float> d((size_t)H * GH, 0.0f);
                if (cell != Cell::GRU) {      // dWh [H, GH] = Hprev^T @ dA
                    pgemm(true, false, H, GH, (int)SB, Hprev.data(), H, dA.data(), GH, 0.0f, d.data(), GH);
                } else {                      // r, z from h; n from r*h
                    const int H2 = 2 * H;
                    pgemm(true, false, H, H2, (int)SB, Hprev.data(), H, dA.data(), GH, 0.0f, d.data(), GH);
                    pgemm(true, false, H, H, (int)SB, rh->data(), H, dA.data() + H2, GH, 0.0f,
                                d.data() + H2, GH);
                }
                unpack_add(d.data(), Wh, H, H, 0, G);
            }
        };
    }

    Result r;
    r.output = slice_of(packed, 0, Shape{S, B, H});
    r.h_n    = slice_of(packed, (int64_t)(S - 1) * BH, Shape{B, H});
    r.c_n    = lstm ? slice_of(packed, SB * H, Shape{B, H}) : nullptr;
    return r;
}

} // namespace rnn_kernel
