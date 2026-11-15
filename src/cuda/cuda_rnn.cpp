// GPU kernels for the recurrent layers (RNN, LSTM, GRU) and the fused MSE
// loss. Split from cuda_ops.cpp; shared helpers in cuda_ops_common.hpp.
#include "cuda_ops_common.hpp"
#include "ml/autograd.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

using namespace std;
using namespace cuda_common;

// ── recurrent layers ─────────────────────────────────────────────────────────
//
// The CPU kernel's scheme on the device: every step's input projection (plus
// bias) in one gemm, then per step one gemm h_{t-1} @ W_h into that step's
// slice and one kernel for the gate arithmetic, which overwrites the slice
// with the activated gates kept for the backward pass. Backward through time
// mirrors it: per step a gate kernel and a gemm back through W_h, then the
// weight gradients over every step at once. Gates are packed side by side,
// [.., G*H]: RNN tanh; LSTM i f o g; GRU r z n.

namespace {

const string& RNN_SRC() {
    static const string s = src(R"(
// sigmoid and tanh on the fast exponential (ex2.approx): the gate kernels
// spent most of their time in five full-precision expf / tanhf per element.
// Within ~1e-7 of the exact functions; tanh saturates cleanly at +-1.
__device__ float sg(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ float th(float x) { return 1.0f - 2.0f / (__expf(2.0f * x) + 1.0f); }
extern "C" __global__ void r_bias_rows(float* y, const float* b, long long rows, int GH) {
    long long i = gid(); if (i >= rows * GH) return;
    y[i] = b[i % GH];
}
// dst [rows, H] = columns [k*H, (k+1)*H) of src [rows, GH]
extern "C" __global__ void r_cols(const float* src, float* dst, long long rows, int H, int GH, int k) {
    long long i = gid(); if (i >= rows * H) return;
    const long long r = i / H; const int j = (int)(i % H);
    dst[i] = src[r * GH + (long long)k * H + j];
}
extern "C" __global__ void r_rnn_fwd(float* a, float* h, int BH) {
    int i = (int)gid(); if (i >= BH) return;
    const float v = th(a[i]); a[i] = v; h[i] = v;
}
extern "C" __global__ void r_lstm_fwd(float* a, const float* c_prev, float* c, float* h, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    float* ar = a + (long long)b * 4 * H;
    const float ig = sg(ar[j]), fg = sg(ar[H + j]), og = sg(ar[2 * H + j]), gg = th(ar[3 * H + j]);
    ar[j] = ig; ar[H + j] = fg; ar[2 * H + j] = og; ar[3 * H + j] = gg;
    const float ct = fg * c_prev[i] + ig * gg;
    c[i] = ct; h[i] = og * th(ct);
}
extern "C" __global__ void r_gru_fwd1(float* a, const float* h_prev, float* rh, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    float* ar = a + (long long)b * 3 * H;
    const float r = sg(ar[j]); ar[j] = r; ar[H + j] = sg(ar[H + j]);
    rh[i] = r * h_prev[i];
}
extern "C" __global__ void r_gru_fwd2(float* a, const float* h_prev, float* h, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    float* ar = a + (long long)b * 3 * H;
    const float n = th(ar[2 * H + j]), z = ar[H + j];
    ar[2 * H + j] = n;
    h[i] = (1.0f - z) * n + z * h_prev[i];
}
// backward: dh holds the gradient arriving from step t+1, dout the output's
extern "C" __global__ void r_rnn_back(const float* g, const float* dout, const float* dh, float* da, int BH) {
    int i = (int)gid(); if (i >= BH) return;
    const float d = dh[i] + dout[i], v = g[i];
    da[i] = d * (1.0f - v * v);
}
extern "C" __global__ void r_lstm_back(const float* g, const float* c, const float* c_prev, const float* dout,
                                       const float* dh, float* dc, float* da, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    const float* gr = g + (long long)b * 4 * H;
    float* dr = da + (long long)b * 4 * H;
    const float ig = gr[j], fg = gr[H + j], og = gr[2 * H + j], gg = gr[3 * H + j];
    const float d = dh[i] + dout[i], tc = th(c[i]);
    const float dct = dc[i] + d * og * (1.0f - tc * tc);
    dr[j]         = dct * gg * ig * (1.0f - ig);
    dr[H + j]     = dct * c_prev[i] * fg * (1.0f - fg);
    dr[2 * H + j] = d * tc * og * (1.0f - og);
    dr[3 * H + j] = dct * ig * (1.0f - gg * gg);
    dc[i] = dct * fg;
}
extern "C" __global__ void r_gru_back1(const float* g, const float* h_prev, const float* dout, const float* dh,
                                       float* da, float* dhp, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    const float* gr = g + (long long)b * 3 * H;
    float* dr = da + (long long)b * 3 * H;
    const float d = dh[i] + dout[i], z = gr[H + j], n = gr[2 * H + j];
    dr[2 * H + j] = d * (1.0f - z) * (1.0f - n * n);
    dr[H + j]     = d * (h_prev[i] - n) * z * (1.0f - z);
    dhp[i] = d * z;
}
extern "C" __global__ void r_gru_back2(const float* g, const float* h_prev, const float* drh, float* da,
                                       float* dhp, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    const float r = g[(long long)b * 3 * H + j];
    da[(long long)b * 3 * H + j] = drh[i] * h_prev[i] * r * (1.0f - r);
    dhp[i] += drh[i] * r;
}
// GRU, PyTorch's form: hh = h W_h for all three gates (no bias), a = the
// input projection plus (b_r, b_z, b_n); a becomes the gates, hn is kept
extern "C" __global__ void r_gruA_fwd(float* a, const float* hh, const float* bhn, const float* h_prev,
                                      float* hn_out, float* h, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    float* ar = a + (long long)b * 3 * H;
    const float* hr = hh + (long long)b * 3 * H;
    const float r = sg(ar[j] + hr[j]), z = sg(ar[H + j] + hr[H + j]);
    const float hn = hr[2 * H + j] + bhn[j];
    const float n = th(ar[2 * H + j] + r * hn);
    ar[j] = r; ar[H + j] = z; ar[2 * H + j] = n;
    hn_out[i] = hn;
    h[i] = (1.0f - z) * n + z * h_prev[i];
}
// its backward: da (input side) and dhh (hidden side, n part dn * r); dh
// becomes d * z, to which the caller adds dhh @ W_h^T
extern "C" __global__ void r_gruA_back(const float* g, const float* h_prev, const float* hn, const float* dout,
                                       float* dh, float* da, float* dhh, int B, int H) {
    int i = (int)gid(); if (i >= B * H) return;
    const int b = i / H, j = i % H;
    const long long o = (long long)b * 3 * H;
    const float r = g[o + j], z = g[o + H + j], n = g[o + 2 * H + j];
    const float d = dh[i] + dout[i];
    const float dn = d * (1.0f - z) * (1.0f - n * n);
    const float dzp = d * (h_prev[i] - n) * z * (1.0f - z);
    const float drp = dn * hn[i] * r * (1.0f - r);
    da[o + j] = drp; da[o + H + j] = dzp; da[o + 2 * H + j] = dn;
    dhh[o + j] = drp; dhh[o + H + j] = dzp; dhh[o + 2 * H + j] = dn * r;
    dh[i] = d * z;
}

// ── the recurrence in one launch ──
// Block x owns hidden units [x*U, x*U + U) -- every gate of them -- for the
// whole batch. Steps are separated by a grid-wide barrier, so every block
// must be resident at once: the host launches no more blocks than
// resident_blocks() allows. Everything a step reads repeatedly is in shared
// memory: the block's columns of W_h and h_{t-1} (forward), its rows of W_h
// and the next step's gradient (backward). bar[0] arrivals, bar[1] generation.
__device__ void grid_sync(unsigned* bar, unsigned nb) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned* vgen = bar + 1;
        const unsigned gen = *vgen;
        __threadfence();
        if (atomicAdd(bar, 1u) == nb - 1) {
            atomicExch(bar, 0u);
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            while (*vgen == gen) { }
        }
        __threadfence();
    }
    __syncthreads();
}
// cell 0 RNN (tanh), 1 LSTM (i f o g), 3 GRU reset-after (r z n). gates holds
// each step's input projection plus bias and becomes the activated gates.
// Shared: ws [H][G*U] (column g*U + uu is W_h's column g*H + u0 + uu),
// hs [B][H], pre [B][G*U] (the step's products).
extern "C" __global__ void r_persist_fwd(int cell, int S, int B, int H, int G, int U, float* gates,
                                         const float* W, const float* h_init, const float* c_init, float* OUT,
                                         float* cs, float* HN, const float* bhn, unsigned* bar) {
    extern __shared__ float sm[];
    const int GH = G * H, BH = B * H, GU = G * U, u0 = blockIdx.x * U;
    float* ws = sm;
    float* hs = ws + H * GU;
    float* pre = hs + BH;
    for (int q = threadIdx.x; q < H * GU; q += blockDim.x) {
        const int k = q / GU, col = q % GU, g = col / U, u = u0 + col % U;
        ws[q] = u < H ? W[(long long)k * GH + g * H + u] : 0.0f;
    }
    for (int t = 0; t < S; t++) {
        const float* hp = t ? OUT + (long long)(t - 1) * BH : h_init;
        if ((BH & 3) == 0 && ((H * GU) & 3) == 0) {
            const float4* h4 = reinterpret_cast<const float4*>(hp);
            float4* s4 = reinterpret_cast<float4*>(hs);
            #pragma unroll 4
            for (int q = threadIdx.x; q < BH / 4; q += blockDim.x) s4[q] = __ldcg(h4 + q);
        } else {
            for (int q = threadIdx.x; q < BH; q += blockDim.x) hs[q] = __ldcg(hp + q);
        }
        __syncthreads();
        // B * G*U products of length H, a thread each: consecutive threads
        // take consecutive columns (no bank conflicts on ws; hs broadcast)
        for (int o = threadIdx.x; o < B * GU; o += blockDim.x) {
            const int b = o / GU, col = o % GU;
            const float* hb = hs + b * H;
            float a0 = 0.0f, a1 = 0.0f;
            int k = 0;
            for (; k + 1 < H; k += 2) { a0 += hb[k] * ws[k * GU + col]; a1 += hb[k + 1] * ws[(k + 1) * GU + col]; }
            if (k < H) a0 += hb[k] * ws[k * GU + col];
            pre[o] = a0 + a1;
        }
        __syncthreads();
        float* ht = OUT + (long long)t * BH;
        for (int o = threadIdx.x; o < B * U; o += blockDim.x) {
            const int b = o / U, uu = o % U, u = u0 + uu;
            if (u >= H) continue;
            const float* pr = pre + b * GU + uu;           // gate g at pr[g * U]
            float* ar = gates + ((long long)t * B + b) * GH;
            const int i = b * H + u;
            if (cell == 0) {
                const float v = th(ar[u] + pr[0]);
                ar[u] = v; ht[i] = v;
            } else if (cell == 1) {
                const float ig = sg(ar[u] + pr[0]), fg = sg(ar[H + u] + pr[U]);
                const float og = sg(ar[2 * H + u] + pr[2 * U]), gg = th(ar[3 * H + u] + pr[3 * U]);
                ar[u] = ig; ar[H + u] = fg; ar[2 * H + u] = og; ar[3 * H + u] = gg;
                const float cp = t ? cs[(long long)(t - 1) * BH + i] : c_init[i];
                const float ct = fg * cp + ig * gg;
                cs[(long long)t * BH + i] = ct;
                ht[i] = og * th(ct);
            } else {
                const float r = sg(ar[u] + pr[0]), z = sg(ar[H + u] + pr[U]);
                const float hn = pr[2 * U] + bhn[u];
                const float n = th(ar[2 * H + u] + r * hn);
                ar[u] = r; ar[H + u] = z; ar[2 * H + u] = n;
                HN[(long long)t * BH + i] = hn;
                ht[i] = (1.0f - z) * n + z * hs[b * H + u];
            }
        }
        if (t + 1 < S) grid_sync(bar, gridDim.x);
    }
}
// Backward through time, the same ownership. dh for unit u is row u of W_h
// against the batch's gradient row from the step after (src: dA, or dHH for
// the GRU): a warp per (batch, unit) product, lanes across the columns, the
// gradient staged through shared memory `chunk` columns at a time.
// Shared: wr [U][GH], sc [B][chunk], acc [B*U].
__device__ void back_products(const float* srow, const float* wr, float* sc, float* acc, int B, int U, int GH,
                              int chunk) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int q = threadIdx.x; q < B * U; q += blockDim.x) acc[q] = 0.0f;
    for (int c0 = 0; c0 < GH; c0 += chunk) {
        const int cn = min(chunk, GH - c0);
        __syncthreads();
        if ((GH & 3) == 0 && (cn & 3) == 0 && (chunk & 3) == 0) {
            const int c4 = cn / 4;
            #pragma unroll 4
            for (int q = threadIdx.x; q < B * c4; q += blockDim.x) {
                const int bb = q / c4, c = q % c4;
                reinterpret_cast<float4*>(sc + bb * chunk)[c] =
                    __ldcg(reinterpret_cast<const float4*>(srow + (long long)bb * GH + c0) + c);
            }
        } else {
            for (int q = threadIdx.x; q < B * cn; q += blockDim.x) {
                const int bb = q / cn, c = q % cn;
                sc[bb * chunk + c] = __ldcg(srow + (long long)bb * GH + c0 + c);
            }
        }
        __syncthreads();
        for (int o = warp; o < B * U; o += nw) {
            const int b = o / U, uu = o % U;
            const float* s = sc + b * chunk;
            const float* w = wr + uu * GH + c0;
            float a = 0.0f;
            for (int c = lane; c < cn; c += 32) a += s[c] * w[c];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
            if (lane == 0) acc[o] += a;
        }
    }
    __syncthreads();
}
extern "C" __global__ void r_persist_bwd(int cell, int S, int B, int H, int G, int U, int chunk,
                                         const float* gates, const float* W, const float* OUT, const float* h_init,
                                         const float* cs, const float* c_init, const float* HN, const float* dOUT,
                                         float* dA, float* dHH, float* dh, float* dc, unsigned* bar) {
    extern __shared__ float sm[];
    const int GH = G * H, BH = B * H, u0 = blockIdx.x * U;
    float* wr = sm;                                  // [U][GH]
    float* acc = wr + U * GH;                        // [B * U], padded to a multiple of 4
    float* sc = acc + ((B * U + 3) & ~3);            // [B][chunk]
    for (int q = threadIdx.x; q < U * GH; q += blockDim.x) {
        const int r = q / GH, c = q % GH;
        wr[q] = u0 + r < H ? W[(long long)(u0 + r) * GH + c] : 0.0f;
    }
    const int idx = threadIdx.x, b = idx / U, uu = idx % U, u = u0 + uu;
    const bool active = idx < B * U && u < H;
    const int i = b * H + u;
    float own = 0.0f;                                // GRU: the part of dh not through W_h
    float dcr = active && cell == 1 ? dc[i] : 0.0f;
    const float* src = cell == 3 ? dHH : dA;
    for (int s = S - 1; s >= 0; s--) {
        // the gradient reaching h_s from step s + 1
        float dht = 0.0f;
        if (s < S - 1) {
            grid_sync(bar, gridDim.x);               // step s + 1 written by every block
            back_products(src + (long long)(s + 1) * B * GH, wr, sc, acc, B, U, GH, chunk);
            if (active) dht = acc[idx] + own;
        }
        if (active) {
            const long long o = ((long long)s * B + b) * GH;
            const float d = dht + dOUT[(long long)s * BH + i];
            if (cell == 0) {
                const float v = gates[o + u];
                dA[o + u] = d * (1.0f - v * v);
            } else if (cell == 1) {
                const float ig = gates[o + u], fg = gates[o + H + u], og = gates[o + 2 * H + u];
                const float gg = gates[o + 3 * H + u];
                const float ct = cs[(long long)s * BH + i];
                const float cp = s ? cs[(long long)(s - 1) * BH + i] : c_init[i];
                const float tc = th(ct);
                const float dct = dcr + d * og * (1.0f - tc * tc);
                dA[o + u]         = dct * gg * ig * (1.0f - ig);
                dA[o + H + u]     = dct * cp * fg * (1.0f - fg);
                dA[o + 2 * H + u] = d * tc * og * (1.0f - og);
                dA[o + 3 * H + u] = dct * ig * (1.0f - gg * gg);
                dcr = dct * fg;
            } else {
                const float r = gates[o + u], z = gates[o + H + u], n = gates[o + 2 * H + u];
                const float hp = s ? OUT[(long long)(s - 1) * BH + i] : h_init[i];
                const float hn = HN[(long long)s * BH + i];
                const float dn = d * (1.0f - z) * (1.0f - n * n);
                const float dzp = d * (hp - n) * z * (1.0f - z);
                const float drp = dn * hn * r * (1.0f - r);
                dA[o + u] = drp; dA[o + H + u] = dzp; dA[o + 2 * H + u] = dn;
                dHH[o + u] = drp; dHH[o + H + u] = dzp; dHH[o + 2 * H + u] = dn * r;
                own = d * z;
            }
        }
    }
    // the gradient for h_0: through step 0's product
    grid_sync(bar, gridDim.x);
    back_products(src, wr, sc, acc, B, U, GH, chunk);
    if (active) {
        dh[i] = acc[idx] + own;
        if (cell == 1) dc[i] = dcr;
    }
}
)");
    return s;
}
cuda::Kernel RK(const char* name) { return K(name, RNN_SRC()); }

// The persistent path's shape: U units per block, one block per slice, the
// batch's h in shared memory (forward), W_h rows and a gradient chunk
// (backward). ok = false when it does not fit or the GPU cannot keep every
// block resident; PROMETHEUS_PERSISTENT_RNN=0 turns it off.
struct Persist {
    bool ok = false;
    int U = 0, blocks = 0, threads = 0, chunk = 0;
    unsigned fwd_shared = 0, bwd_shared = 0;
};
Persist persist_plan(int cell, int B, int H, int G) {
    Persist p;
    // where it is the faster path (A100, 64 x 32, 128 -> 256): RNN 2.3 ms in
    // one launch vs 2.6 per step; LSTM 4.8 vs 3.3 and GRU 3.8 vs 3.4 -- there
    // every block re-reads the whole batch's 4H- / 3H-wide gradient from L2
    // each step. PROMETHEUS_PERSISTENT_RNN=1 forces it for every cell, =0 off.
    static const int mode = [] {
        const char* e = std::getenv("PROMETHEUS_PERSISTENT_RNN");
        return !e || !*e ? -1 : e[0] == '1' ? 1 : 0;
    }();
    const bool on = mode == 1 || (mode == -1 && cell == 0);
    const int sms = cuda::multiprocessor_count();
    if (!on || cell == 2 || sms <= 0 || B <= 0 || H <= 0) return p;
    const size_t limit = 48 * 1024;
    p.U = (H + sms - 1) / sms;
    p.blocks = (H + p.U - 1) / p.U;
    if (B * p.U > 1024) return p;
    p.threads = std::max(512, (B * p.U + 31) / 32 * 32);
    const size_t GH = (size_t)G * H, GU = (size_t)G * p.U, f = sizeof(float);
    const size_t fsh = ((size_t)H * GU + (size_t)B * H + (size_t)B * GU) * f;
    const size_t fixed = ((size_t)p.U * GH + (((size_t)B * p.U + 3) & ~(size_t)3)) * f;
    if (fsh > limit || fixed + (size_t)B * 32 * f > limit) return p;
    p.chunk = (int)std::min<size_t>(GH, (limit - fixed) / ((size_t)B * f) / 32 * 32);
    if (p.chunk < 32) return p;
    p.fwd_shared = (unsigned)fsh;
    p.bwd_shared = (unsigned)(fixed + (size_t)B * p.chunk * f);
    if (cuda::resident_blocks(RK("r_persist_fwd"), p.threads, p.fwd_shared) < p.blocks ||
        cuda::resident_blocks(RK("r_persist_bwd"), p.threads, p.bwd_shared) < p.blocks)
        return p;
    p.ok = true;
    return p;
}

const float* cf(const TensorPtr& t) { return static_cast<const float*>(base(t)); }
float* mf(const TensorPtr& t) { return static_cast<float*>(t->storage()->data()); }

// [rows, G*H] -> G tensors [rows, H]
vector<TensorPtr> split_cols(const TensorPtr& src, long long rows, int H, int G) {
    vector<TensorPtr> out;
    for (int k = 0; k < G; k++) {
        auto d = empty_cuda({rows, (long long)H});
        const void* ps = base(src);
        void* pd = d->storage()->data();
        long long rr = rows;
        int h = H, gh = G * H, kk = k;
        void* a[] = {&ps, &pd, &rr, &h, &gh, &kk};
        cuda::launch(RK("r_cols"), rows * H, a);
        out.push_back(d);
    }
    return out;
}

} // namespace

namespace cuda_ops {

TensorPtr rnn(int cell, const TensorPtr& input0, const TensorPtr& h00, const TensorPtr& c00, int H,
              const vector<TensorPtr>& Wx0, const vector<TensorPtr>& Wh0, const vector<TensorPtr>& b0,
              Reference reference) {
    const bool lstm = cell == 1, gru = cell == 2, after = cell == 3;
    auto X = contiguous_on_device(f32(input0, "RNN"));
    const int S = (int)X->size(0), B = (int)X->size(1), I = (int)X->size(2);
    const int G = (int)Wx0.size(), GH = G * H, BH = B * H;
    const long long SB = (long long)S * B;
    TensorPtr h0 = h00 ? contiguous_on_device(f32(h00, "RNN")) : nullptr;
    TensorPtr c0 = (lstm && c00) ? contiguous_on_device(f32(c00, "RNN")) : nullptr;
    TensorPtr Wxp, Whp, WhpT, bp, bhn;
    {
        NoGrad off;
        vector<TensorPtr> wx, wh, bb;
        for (auto& w : Wx0) wx.push_back(f32(w, "RNN"));
        for (auto& w : Wh0) wh.push_back(f32(w, "RNN"));
        for (size_t k = 0; k < (after ? 3 : b0.size()); k++) bb.push_back(f32(b0[k], "RNN"));
        if (after) bhn = contiguous_on_device(f32(b0[3], "RNN"));
        Wxp = contiguous(concat_op(wx, 1));                    // [I, GH]
        Whp = contiguous(concat_op(wh, 1));                    // [H, GH]
        // the backward's products with W_h^T as plain NN gemms against a copy
        // laid out [GH, H]: cuBLAS's NT kernel for 32 x 256 x 256 took 15 us
        // a step against 9 us for NN on a GTX 970
        WhpT = contiguous(transpose(Whp, 0, 1));
        bp  = contiguous(concat_op(bb, 0));                    // [GH]
    }
    auto zeros_bh = make_shared<Tensor>(Shape{(long long)BH}, F32, Device::CUDA);   // zeroed
    const float* h_init = h0 ? cf(h0) : cf(zeros_bh);
    const float* c_init = c0 ? cf(c0) : cf(zeros_bh);

    // 1. every step's input projection plus bias; becomes the activated gates
    auto gates = empty_cuda({SB, (long long)GH});
    {
        const void* pb = base(bp);
        void* pg = gates->storage()->data();
        long long rows = SB;
        int gh = GH;
        void* a[] = {&pg, &pb, &rows, &gh};
        cuda::launch(RK("r_bias_rows"), SB * GH, a);
        cuda::sgemm(false, false, (int)SB, GH, I, 1.0f, cf(X), I, cf(Wxp), GH, 1.0f, mf(gates), GH);
    }
    // outputs: h for every step, then c_n
    auto packed = empty_cuda({SB * H + (lstm ? BH : 0)});
    auto cs = lstm ? empty_cuda({SB, (long long)H}) : nullptr;
    auto rh = gru ? empty_cuda({SB, (long long)H}) : nullptr;
    // reset-after GRU: hn = h W_hn + b_hn per step, and a step's h W_h
    auto HN = after ? empty_cuda({SB, (long long)H}) : nullptr;
    auto hh = after ? empty_cuda({(long long)B, (long long)GH}) : nullptr;
    float* OUT = mf(packed);
    int b_ = B, h_ = H, bh = BH;

    // 2. the recurrence: in one launch where it fits (a launch a step left
    // the fast GPUs waiting on the CPU), else a gemm and a kernel a step
    const Persist plan = persist_plan(cell, B, H, G);
    if (plan.ok && S > 0) {
        auto bar = make_shared<Tensor>(Shape{2}, DType::Int32, Device::CUDA);          // zeroed
        int c_ = cell, s_ = S, g_ = G, u_ = plan.U;
        float* pg = mf(gates);
        const float* pw = cf(Whp);
        float* pcs = lstm ? mf(cs) : nullptr;
        float* phn = after ? mf(HN) : nullptr;
        const float* pbn = after ? cf(bhn) : nullptr;
        void* pbar = bar->storage()->data();
        void* a[] = {&c_, &s_, &b_, &h_, &g_, &u_, &pg, &pw, &h_init, &c_init, &OUT, &pcs, &phn, &pbn, &pbar};
        cuda::launch(RK("r_persist_fwd"), (unsigned)plan.blocks, 1, 1, (unsigned)plan.threads, 1, 1, plan.fwd_shared, a);
    }
    for (int t = 0; t < (plan.ok ? 0 : S); t++) {
        const float* hp = t ? OUT + (size_t)(t - 1) * BH : h_init;
        float* at = mf(gates) + (size_t)t * B * GH;
        float* ht = OUT + (size_t)t * BH;
        if (cell == 0) {
            cuda::sgemm(false, false, B, H, H, 1.0f, hp, H, cf(Whp), GH, 1.0f, at, GH);
            void* a[] = {&at, &ht, &bh};
            cuda::launch(RK("r_rnn_fwd"), BH, a);
        } else if (lstm) {
            cuda::sgemm(false, false, B, GH, H, 1.0f, hp, H, cf(Whp), GH, 1.0f, at, GH);
            const float* cp = t ? mf(cs) + (size_t)(t - 1) * BH : c_init;
            float* ct = mf(cs) + (size_t)t * BH;
            void* a[] = {&at, &cp, &ct, &ht, &b_, &h_};
            cuda::launch(RK("r_lstm_fwd"), BH, a);
        } else if (after) {
            // one product for all three gates
            float* phh = mf(hh);
            cuda::sgemm(false, false, B, GH, H, 1.0f, hp, H, cf(Whp), GH, 0.0f, phh, GH);
            const float* pbn = cf(bhn);
            float* hnt = mf(HN) + (size_t)t * BH;
            void* a[] = {&at, &phh, &pbn, &hp, &hnt, &ht, &b_, &h_};
            cuda::launch(RK("r_gruA_fwd"), BH, a);
        } else {
            cuda::sgemm(false, false, B, 2 * H, H, 1.0f, hp, H, cf(Whp), GH, 1.0f, at, GH);
            float* rt = mf(rh) + (size_t)t * BH;
            void* a1[] = {&at, &hp, &rt, &b_, &h_};
            cuda::launch(RK("r_gru_fwd1"), BH, a1);
            cuda::sgemm(false, false, B, H, H, 1.0f, rt, H, cf(Whp) + 2 * H, GH, 1.0f, at + 2 * H, GH);
            void* a2[] = {&at, &hp, &ht, &b_, &h_};
            cuda::launch(RK("r_gru_fwd2"), BH, a2);
        }
    }
    if (lstm) cuda::copy_device(OUT + SB * H, mf(cs) + (size_t)(S - 1) * BH, (size_t)BH * sizeof(float));

    vector<TensorPtr> inputs = {input0};
    if (h00) inputs.push_back(h00);
    if (lstm && c00) inputs.push_back(c00);
    for (auto* group : {&Wx0, &Wh0, &b0}) for (auto& w : *group) inputs.push_back(w);

    attach_fused(packed, inputs, (gru || after) ? "GRU" : lstm ? "LSTM" : "RNN", reference,
                 [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto Gd = contiguous_on_device(f32(G0, "RNN"));
        const float* dOUT = cf(Gd);
        const float* OUTd = cf(packed);
        const float* hi = h0 ? cf(h0) : cf(zeros_bh);
        const float* ci = c0 ? cf(c0) : cf(zeros_bh);
        auto dA = empty_cuda({SB, (long long)GH});
        auto dh = make_shared<Tensor>(Shape{(long long)BH}, F32, Device::CUDA);       // zeroed
        auto dc = lstm ? empty_cuda({(long long)BH}) : nullptr;
        auto dhp = gru ? empty_cuda({(long long)BH}) : nullptr;
        auto drh = gru ? empty_cuda({(long long)BH}) : nullptr;
        auto dHH = after ? empty_cuda({SB, (long long)GH}) : nullptr;
        if (lstm) cuda::copy_device(mf(dc), dOUT + SB * H, (size_t)BH * sizeof(float));
        int b2 = B, h2 = H, bh2 = BH;

        // 3. backward through time: one launch, or a kernel and a gemm a step
        if (plan.ok && S > 0) {
            auto bar = make_shared<Tensor>(Shape{2}, DType::Int32, Device::CUDA);      // zeroed
            int c_ = cell, s_ = S, g_ = G, u_ = plan.U, ch = plan.chunk;
            const float* pg = cf(gates);
            const float* pw = cf(Whp);
            const float* pcs = lstm ? cf(cs) : nullptr;
            const float* phn = after ? cf(HN) : nullptr;
            float* pda = mf(dA);
            float* pdq = after ? mf(dHH) : nullptr;
            float* pdh = mf(dh);
            float* pdc = lstm ? mf(dc) : nullptr;
            void* pbar = bar->storage()->data();
            void* a[] = {&c_, &s_, &b2, &h2, &g_, &u_, &ch, &pg, &pw, &OUTd, &hi, &pcs, &ci, &phn, &dOUT,
                         &pda, &pdq, &pdh, &pdc, &pbar};
            cuda::launch(RK("r_persist_bwd"), (unsigned)plan.blocks, 1, 1, (unsigned)plan.threads, 1, 1,
                         plan.bwd_shared, a);
        }
        for (int t = (plan.ok ? -1 : S - 1); t >= 0; t--) {
            const float* hp = t ? OUTd + (size_t)(t - 1) * BH : hi;
            const float* gt = cf(gates) + (size_t)t * B * GH;
            float* da = mf(dA) + (size_t)t * B * GH;
            const float* dout = dOUT + (size_t)t * BH;
            float* pdh = mf(dh);
            if (cell == 0) {
                void* a[] = {&gt, &dout, &pdh, &da, &bh2};
                cuda::launch(RK("r_rnn_back"), BH, a);
                cuda::sgemm(false, false, B, H, GH, 1.0f, da, GH, cf(WhpT), H, 0.0f, pdh, H);
            } else if (lstm) {
                const float* ct = cf(cs) + (size_t)t * BH;
                const float* cp = t ? cf(cs) + (size_t)(t - 1) * BH : ci;
                float* pdc = mf(dc);
                void* a[] = {&gt, &ct, &cp, &dout, &pdh, &pdc, &da, &b2, &h2};
                cuda::launch(RK("r_lstm_back"), BH, a);
                cuda::sgemm(false, false, B, H, GH, 1.0f, da, GH, cf(WhpT), H, 0.0f, pdh, H);
            } else if (after) {
                const float* hnt = cf(HN) + (size_t)t * BH;
                float* dq = mf(dHH) + (size_t)t * B * GH;
                void* a[] = {&gt, &hp, &hnt, &dout, &pdh, &da, &dq, &b2, &h2};
                cuda::launch(RK("r_gruA_back"), BH, a);
                // dh = d * z + dhh @ W_h^T, one product
                cuda::sgemm(false, false, B, H, GH, 1.0f, dq, GH, cf(WhpT), H, 1.0f, pdh, H);
            } else {
                float* pp = mf(dhp);
                float* pr = mf(drh);
                void* a1[] = {&gt, &hp, &dout, &pdh, &da, &pp, &b2, &h2};
                cuda::launch(RK("r_gru_back1"), BH, a1);
                // through (r * h) @ W_hn
                cuda::sgemm(false, false, B, H, H, 1.0f, da + 2 * H, GH, cf(WhpT) + (size_t)2 * H * H, H, 0.0f, pr, H);
                void* a2[] = {&gt, &hp, &pr, &da, &pp, &b2, &h2};
                cuda::launch(RK("r_gru_back2"), BH, a2);
                // through h @ [W_hr W_hz]
                cuda::sgemm(false, false, B, H, 2 * H, 1.0f, da, GH, cf(WhpT), H, 1.0f, pp, H);
                cuda::copy_device(pdh, pp, (size_t)BH * sizeof(float));
            }
        }

        // 4. over every step at once
        vector<TensorPtr> out;
        TensorPtr dX;
        if (input0->requires_grad()) {
            dX = empty_cuda({(long long)S, (long long)B, (long long)I});
            cuda::sgemm(false, true, (int)SB, I, GH, 1.0f, cf(dA), GH, cf(Wxp), GH, 0.0f, mf(dX), I);
        }
        out.push_back(dX);
        if (h00) out.push_back(reshape(dh, {(long long)B, (long long)H}));
        if (lstm && c00) out.push_back(reshape(dc, {(long long)B, (long long)H}));
        auto dWx = empty_cuda({(long long)I, (long long)GH});
        cuda::sgemm(true, false, I, GH, (int)SB, 1.0f, cf(X), I, cf(dA), GH, 0.0f, mf(dWx), GH);
        for (auto& d : split_cols(dWx, I, H, G)) out.push_back(d);
        // h_{t-1} for every step: h0, then the outputs shifted by one
        auto Hprev = empty_cuda({SB, (long long)H});
        cuda::copy_device(mf(Hprev), hi, (size_t)BH * sizeof(float));
        if (S > 1) cuda::copy_device(mf(Hprev) + BH, OUTd, (size_t)(S - 1) * BH * sizeof(float));
        auto dWh = empty_cuda({(long long)H, (long long)GH});
        if (after) {
            cuda::sgemm(true, false, H, GH, (int)SB, 1.0f, cf(Hprev), H, cf(dHH), GH, 0.0f, mf(dWh), GH);
        } else if (!gru) {
            cuda::sgemm(true, false, H, GH, (int)SB, 1.0f, cf(Hprev), H, cf(dA), GH, 0.0f, mf(dWh), GH);
        } else {
            cuda::sgemm(true, false, H, 2 * H, (int)SB, 1.0f, cf(Hprev), H, cf(dA), GH, 0.0f, mf(dWh), GH);
            cuda::sgemm(true, false, H, H, (int)SB, 1.0f, cf(rh), H, cf(dA) + 2 * H, GH, 0.0f, mf(dWh) + 2 * H, GH);
        }
        for (auto& d : split_cols(dWh, H, H, G)) out.push_back(d);
        auto db = ::sum(dA, 0);                                                // [GH]
        for (auto& d : split_cols(db, 1, H, G)) out.push_back(reshape(d, {(long long)H}));
        if (after) {                                                           // b_hn: dhh's n part
            auto dbh = ::sum(dHH, 0);
            out.push_back(reshape(split_cols(dbh, 1, H, G)[2], {(long long)H}));
        }
        return out;
    });
    return packed;
}

} // namespace cuda_ops

// ── MSE as one kernel each way ───────────────────────────────────────────────
//
// mean((p - t)^2) was subtract, pow, mean and their backwards: ~10 small
// kernels, ~0.4 ms of an LSTM step on a GTX 970. Forward: block partials of
// (p - t)^2 / n added into one float. Backward: both gradients in one pass.

namespace {
const string& MSE_SRC() {
    static const string s = src(R"(
extern "C" __global__ void l_mse(const float* p, const float* t, float* out, long long n, float inv_n) {
    __shared__ float sh[256];
    float acc = 0.0f;
    for (long long i = gid(); i < n; i += (long long)gridDim.x * blockDim.x) { const float d = p[i] - t[i]; acc += d * d; }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) { if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(out, sh[0] * inv_n);
}
// dp = 2 (p - t) G / n, dt = -dp (either may be null)
extern "C" __global__ void b_mse(const float* p, const float* t, const float* G, float* dp, float* dt, long long n,
                                 float inv_n) {
    long long i = gid(); if (i >= n) return;
    const float g = 2.0f * (p[i] - t[i]) * G[0] * inv_n;
    if (dp) dp[i] = g;
    if (dt) dt[i] = -g;
}
)");
    return s;
}
} // namespace

namespace cuda_ops {

TensorPtr mse(const TensorPtr& p0, const TensorPtr& t0, Reference reference) {
    auto p = contiguous_on_device(f32(p0, "mse_loss")), t = contiguous_on_device(f32(t0, "mse_loss"));
    long long n = p->numel();
    auto out = make_shared<Tensor>(Shape{1}, F32, Device::CUDA);              // zeroed
    float inv = n ? 1.0f / (float)n : 0.0f;
    if (n) {
        const void* pp = base(p);
        const void* pt = base(t);
        void* po = out->storage()->data();
        const unsigned blocks = (unsigned)std::min<long long>((n + 255) / 256, 1024);
        void* a[] = {&pp, &pt, &po, &n, &inv};
        cuda::launch(K("l_mse", MSE_SRC()), blocks, 1, 1, 256, 1, 1, 0, a);
    }
    attach_fused(out, {p0, t0}, "mse_loss", reference, [=](const TensorPtr& G0) -> vector<TensorPtr> {
        auto G = contiguous_on_device(f32(G0, "mse_loss"));
        TensorPtr dp = p0->requires_grad() ? empty_cuda(p->shape()) : nullptr;
        TensorPtr dt = t0->requires_grad() ? empty_cuda(t->shape()) : nullptr;
        long long nn = n;
        float iv = inv;
        if (nn && (dp || dt)) {
            const void* pp = base(p);
            const void* pt = base(t);
            const void* pg = base(G);
            void* pdp = dp ? dp->storage()->data() : nullptr;
            void* pdt = dt ? dt->storage()->data() : nullptr;
            void* a[] = {&pp, &pt, &pg, &pdp, &pdt, &nn, &iv};
            cuda::launch(K("b_mse", MSE_SRC()), nn, a);
        }
        return {dp, dt};
    });
    return out;
}

} // namespace cuda_ops
