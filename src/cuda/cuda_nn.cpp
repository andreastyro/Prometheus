// GPU kernels for the network layers: LayerNorm / GELU / embedding backward,
// fused attention (RoPE, one QKV gemm) and the transformer block's fused pieces.
// Split from cuda_ops.cpp so the parts compile in parallel; shared helpers in
// cuda_ops_common.hpp.
#include "cuda_ops_common.hpp"
#include "ml/autograd.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

using namespace std;
using namespace cuda_common;

namespace cuda_ops {

// ── fused backward kernels ───────────────────────────────────────────────────
//
// A GPU op's backward_graph builds its gradient from ops, so it can be
// differentiated again -- but for LayerNorm, GELU and the embedding that is a
// dozen or more small kernels per call, most of the backward's time. When the
// gradient is not itself being recorded (a plain .backward(), or grad()
// without create_graph -- grad_enabled() is false), these fused kernels give
// the same numbers in one or two launches; with create_graph the op form runs.

static const string& BACKWARD_SRC() {
    static const string s = src(R"(
// LayerNorm backward, one warp per row (8 rows to a block), sums by shuffles:
// dx, and the row's mean and 1/std for the gamma / beta column sums.
//   dx = inv * (g*dy - mean(g*dy) - xhat * mean(g*dy*xhat))
// A block per row spent most of its time in __syncthreads on rows of a few
// hundred: 2x PyTorch's time on a GTX 970.
__device__ float ln_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
extern "C" __global__ void b_layernorm_rows(const float* x, const float* g, const float* dy, float* dx,
                                            float* mean_out, float* inv_out, long long cols, float eps,
                                            long long rows) {
    const long long r = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int lane = threadIdx.x & 31;
    const float* xr = x + r * cols;
    const float* dyr = dy + r * cols;
    float* dxr = dx + r * cols;
    float t = 0.0f;
    for (long long j = lane; j < cols; j += 32) t += xr[j];
    const float mean = ln_warp_sum(t) / (float)cols;
    t = 0.0f;
    for (long long j = lane; j < cols; j += 32) { const float d = xr[j] - mean; t += d * d; }
    const float inv = rsqrtf(ln_warp_sum(t) / (float)cols + eps);
    float a = 0.0f, b = 0.0f;
    for (long long j = lane; j < cols; j += 32) {
        const float gd = g[j] * dyr[j];
        a += gd;
        b += gd * (xr[j] - mean) * inv;
    }
    const float m1 = ln_warp_sum(a) / (float)cols, m2 = ln_warp_sum(b) / (float)cols;
    for (long long j = lane; j < cols; j += 32) {
        const float xh = (xr[j] - mean) * inv;
        dxr[j] = inv * (g[j] * dyr[j] - m1 - xh * m2);
    }
    if (lane == 0) { mean_out[r] = mean; inv_out[r] = inv; }
}
// The same with x and dy held in registers (rows of up to 1024), and the
// gamma / beta sums done here too: each block folds its rows' dy * xhat and dy
// into shared memory and adds one value per column (dg, db start at zero) --
// the separate column pass read x and dy a second time. Each warp takes 4
// rows, so there are as many atomics per column as the old chunked pass had.
// accum: dx += instead of dx = (the input's gradient buffer, already holding
// another branch's gradient: no separate add pass).
extern "C" __global__ void b_layernorm_rows_w(const float* x, const float* g, const float* dy, float* dx,
                                              int cols, float eps, long long rows, int accum,
                                              float* dg, float* db) {
    __shared__ float sg[1024], sb[1024];
    for (int j = threadIdx.x; j < cols; j += blockDim.x) { sg[j] = 0.0f; sb[j] = 0.0f; }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int q = 0; q < 4; q++) {
        const long long r = ((long long)blockIdx.x * 8 + warp) * 4 + q;
        if (r >= rows) break;
        const float* xr = x + r * cols;
        const float* dyr = dy + r * cols;
        float xv[32], dv[32];
        float t = 0.0f;
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            const int j = lane + 32 * k;
            xv[k] = j < cols ? xr[j] : 0.0f;
            dv[k] = j < cols ? dyr[j] : 0.0f;
            t += xv[k];
        }
        const float mean = ln_warp_sum(t) / (float)cols;
        t = 0.0f;
        #pragma unroll
        for (int k = 0; k < 32; k++) { const int j = lane + 32 * k; if (j < cols) { const float d = xv[k] - mean; t += d * d; } }
        const float inv = rsqrtf(ln_warp_sum(t) / (float)cols + eps);
        float a = 0.0f, b = 0.0f;
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            const int j = lane + 32 * k;
            if (j < cols) {
                const float xh = (xv[k] - mean) * inv, gd = g[j] * dv[k];
                a += gd;
                b += gd * xh;
                atomicAdd(&sg[j], dv[k] * xh);
                atomicAdd(&sb[j], dv[k]);
            }
        }
        const float m1 = ln_warp_sum(a) / (float)cols, m2 = ln_warp_sum(b) / (float)cols;
        float* dxr = dx + r * cols;
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            const int j = lane + 32 * k;
            if (j < cols) {
                const float v = inv * (g[j] * dv[k] - m1 - (xv[k] - mean) * inv * m2);
                dxr[j] = accum ? dxr[j] + v : v;
            }
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < cols; j += blockDim.x) { atomicAdd(&dg[j], sg[j]); atomicAdd(&db[j], sb[j]); }
}
// gamma / beta gradients: a thread per (chunk of 32 rows, column), added
// atomically -- one thread per column left most of the GPU idle
extern "C" __global__ void b_layernorm_cols(const float* x, const float* dy, const float* mean, const float* inv,
                                            float* dg, float* db, long long rows, long long cols) {
    const long long chunks = (rows + 31) / 32;
    long long i = gid(); if (i >= cols * chunks) return;
    const long long j = i % cols, r0 = (i / cols) * 32, r1 = min(rows, r0 + 32);
    float ag = 0.0f, ab = 0.0f;
    for (long long r = r0; r < r1; r++) {
        const float d = dy[r * cols + j];
        ag += d * (x[r * cols + j] - mean[r]) * inv[r];
        ab += d;
    }
    atomicAdd(&dg[j], ag);
    atomicAdd(&db[j], ab);
}
// GELU (tanh form): dx = dy * d/dx [0.5 x (1 + tanh(k (x + c x^3)))]
extern "C" __global__ void b_gelu(const float* x, const float* dy, float* dx, long long n) {
    long long i = gid(); if (i >= n) return;
    const float k = 0.7978845608f, c = 0.044715f, v = x[i];
    const float t = tanhf(k * (v + c * v * v * v));
    dx[i] = dy[i] * (0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * k * (1.0f + 3.0f * c * v * v));
}
// embedding: rows of the table by id, and the scatter-add back (ids repeat).
// ids come from a device tensor (dtype code dt) or, with mod > 0, are the
// positions r % mod. Checking range on the host would mean reading the ids
// back every step; an id outside [0, vocab) reads as a zero row instead.
__device__ long long emb_id(const void* ids, int dt, long long r, long long mod) {
    if (mod > 0) return r % mod;
    return dt == 5 ? (long long)((const int*)ids)[r] : ((const long long*)ids)[r];
}
extern "C" __global__ void b_embed_fwd(const float* w, const void* ids, int dt, long long mod, float* out,
                                       long long rows, long long e, long long vocab) {
    long long i = gid(); if (i >= rows * e) return;
    const long long id = emb_id(ids, dt, i / e, mod);
    out[i] = (id < 0 || id >= vocab) ? 0.0f : w[id * e + i % e];
}
extern "C" __global__ void b_embed_bwd(const float* g, const void* ids, int dt, long long mod, float* dw,
                                       long long rows, long long e, long long vocab) {
    long long i = gid(); if (i >= rows * e) return;
    const long long id = emb_id(ids, dt, i / e, mod);
    if (id >= 0 && id < vocab) atomicAdd(&dw[id * e + i % e], g[i]);
}
)");
    return s;
}

vector<TensorPtr> layernorm_backward(const TensorPtr& x0, const TensorPtr& g0, const TensorPtr& G0, float eps,
                                     const TensorPtr& add_to) {
    auto x = contiguous_on_device(f32(x0, "LayerNorm")), g = contiguous_on_device(f32(g0, "LayerNorm"));
    auto G = contiguous_on_device(f32(G0, "LayerNorm"));
    long long cols = x->shape().back();
    long long rows = cols ? x->numel() / cols : 0;
    // add_to: x's gradient buffer already holding another branch's gradient
    // (the residual stream): dx is added into it and none is returned
    TensorPtr into = add_to && add_to->numel() == x->numel() && add_to->is_contiguous() ? add_to : nullptr;
    // the register kernel adds into the buffer itself; the wide-row path
    // computes dx fresh and adds it after
    auto dx = into && cols <= 1024
                  ? make_shared<Tensor>(into->storage(), x->shape(), Tensor::contiguous_strides(x->shape()),
                                        into->offset())
                  : empty_cuda(x->shape());
    auto dg = make_shared<Tensor>(Shape{cols}, F32, Device::CUDA), db = make_shared<Tensor>(Shape{cols}, F32, Device::CUDA);
    auto mean = empty_cuda({max(rows, 1LL)}), inv = empty_cuda({max(rows, 1LL)});
    const void* px = base(x);
    const void* pg = base(g);
    const void* pd = base(G);
    void* pdx = dx->storage()->data();
    void* pm = mean->storage()->data();
    void* pi = inv->storage()->data();
    void* pdg = dg->storage()->data();
    void* pdb = db->storage()->data();
    if (rows && cols <= 1024) {
        // dx, and dgamma / dbeta, in one kernel
        int cc = (int)cols, ac = into ? 1 : 0;
        void* aw[] = {&px, &pg, &pd, &pdx, &cc, &eps, &rows, &ac, &pdg, &pdb};
        cuda::launch(K("b_layernorm_rows_w", BACKWARD_SRC()), (unsigned)((rows + 31) / 32), 1, 1, 256, 1, 1, 0, aw);
        return {into ? nullptr : dx, dg, db};
    }
    void* a1[] = {&px, &pg, &pd, &pdx, &pm, &pi, &cols, &eps, &rows};
    if (rows) cuda::launch(K("b_layernorm_rows", BACKWARD_SRC()), (unsigned)((rows + 7) / 8), 1, 1, 256, 1, 1, 0, a1);
    void* a2[] = {&px, &pd, &pm, &pi, &pdg, &pdb, &rows, &cols};
    cuda::launch(K("b_layernorm_cols", BACKWARD_SRC()), cols * ((rows + 31) / 32), a2);
    if (into) {                       // wide rows: the separate add
        NoGrad off;
        accumulate(into->storage()->data(), into->numel(), dx);
        return {nullptr, dg, db};
    }
    return {dx, dg, db};
}

TensorPtr gelu_backward(const TensorPtr& x0, const TensorPtr& G0) {
    auto x = contiguous_on_device(f32(x0, "gelu")), G = contiguous_on_device(f32(G0, "gelu"));
    auto dx = empty_cuda(x->shape());
    const void* px = base(x);
    const void* pg = base(G);
    void* pd = dx->storage()->data();
    long long n = x->numel();
    void* args[] = {&px, &pg, &pd, &n};
    cuda::launch(K("b_gelu", BACKWARD_SRC()), n, args);
    return dx;
}

// Rows of W by id. `ids` is an int64 / int32 tensor on the device, or null
// with mod > 0 for positions r % mod. The op form (for recorded gradients)
// reads the ids back; a plain forward and backward never do.
TensorPtr embedding_device(const TensorPtr& W0, const TensorPtr& ids, long long mod, long long rows,
                           const Shape& out_shape) {
    auto W = contiguous_on_device(f32(W0, "Embedding"));
    TensorPtr dev_ids = ids ? contiguous_on_device(on_device(ids, "Embedding")) : nullptr;
    if (dev_ids && dev_ids->dtype() != DType::Int64 && dev_ids->dtype() != DType::Int32)
        dev_ids = cast_op(dev_ids, DType::Int64);
    long long e = W->shape().back(), vocab = W->shape()[0];
    auto out = empty_cuda(out_shape);
    const void* pw = base(W);
    const void* pi = dev_ids ? base(dev_ids) : nullptr;
    int dt = dev_ids ? (int)dev_ids->dtype() : 4;
    void* po = out->storage()->data();
    long long r = rows, md = mod;
    void* args[] = {&pw, &pi, &dt, &md, &po, &r, &e, &vocab};
    cuda::launch(K("b_embed_fwd", BACKWARD_SRC()), rows * e, args);
    if (W->requires_grad()) {
        auto node = make_node(out, {W});
        node->name = "Embedding";
        // the op form, for gradients that are themselves differentiated
        autograd::attach_reference(*node, [dev_ids, mod, rows, e, out_shape](const vector<TensorPtr>& in) -> TensorPtr {
            vector<int64_t> host((size_t)rows);
            if (dev_ids) host = index_values(dev_ids, "Embedding");
            else for (long long q = 0; q < rows; q++) host[(size_t)q] = q % mod;
            auto idx = make_shared<vector<int64_t>>((size_t)(rows * e));
            for (long long q = 0; q < rows; q++)
                for (long long d = 0; d < e; d++) (*idx)[(size_t)(q * e + d)] = host[(size_t)q] * e + d;
            return gather_flat(in[0], idx, out_shape);
        });
        auto reference = node->backward_graph;
        node->backward_graph = [reference, dev_ids, dt, mod, rows, e, vocab, W](const TensorPtr& out, const TensorPtr& G0) -> vector<TensorPtr> {
            if (grad_enabled()) return reference(out, G0);
            auto G = contiguous_on_device(f32(G0, "Embedding"));
            auto dw = make_shared<Tensor>(W->shape(), F32, Device::CUDA);      // zeroed
            const void* pg = base(G);
            const void* pi = dev_ids ? base(dev_ids) : nullptr;
            void* pd = dw->storage()->data();
            int d2 = dt;
            long long r2 = rows, ee = e, md2 = mod, vv = vocab;
            void* a[] = {&pg, &pi, &d2, &md2, &pd, &r2, &ee, &vv};
            cuda::launch(K("b_embed_bwd", BACKWARD_SRC()), r2 * ee, a);
            return {dw};
        };
    }
    return out;
}

TensorPtr embedding(const TensorPtr& W0, const vector<int64_t>& ids, const Shape& out_shape) {
    auto host = make_shared<Tensor>(Shape{(int64_t)ids.size()}, DType::Int64);
    if (!ids.empty()) memcpy(host->storage()->data(), ids.data(), ids.size() * sizeof(int64_t));
    return embedding_device(W0, host->to(Device::CUDA), 0, (long long)ids.size(), out_shape);
}

// ── fused attention ──────────────────────────────────────────────────────────
//
// softmax(scale * Q K^T + causal mask) V for every head, with Q, K, V and the
// output in [B, S, E] -- heads side by side in each row, as the projections
// produce them. Head h of sequence b starts at b*S*E + h*D with a row stride
// of E, and the H heads of one sequence are a strided batch of stride D, so
// cuBLAS reads them in place: no permute, no copies. Per sequence: one batched
// gemm for the scores, one kernel for scale + mask + softmax, one batched gemm
// for P V. The backward is four batched gemms and one kernel for the softmax
// gradient. With create_graph the op form runs instead.

static const string& ATTN_SRC() {
    static const string s = src(R"(
// rows of S scores: P = softmax(scale * s), with j > i masked when causal; in place
// One warp per row (8 rows to a 256-thread block), reductions by shuffles:
// a block per row left most of its threads idle on rows of a few hundred and
// spent its time in __syncthreads -- 4x slower than PyTorch's on a GTX 970.
__device__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// rows of S scores: P = softmax(scale * s), with j > i masked when causal; in place
extern "C" __global__ void a_softmax(float* sc, long long cols, float scale, int causal, long long rows) {
    const long long r = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int lane = threadIdx.x & 31;
    float* row = sc + r * cols;
    const long long i = r % cols;                       // the query's position
    const long long last = causal ? i : cols - 1;
    float m = -3.402823466e38f;
    for (long long j = lane; j <= last; j += 32) m = fmaxf(m, row[j] * scale);
    m = warp_max(m);
    float t = 0.0f;
    for (long long j = lane; j < cols; j += 32) {
        const float e = j <= last ? expf(row[j] * scale - m) : 0.0f;
        row[j] = e;
        t += e;
    }
    const float inv = 1.0f / warp_sum(t);
    for (long long j = lane; j < cols; j += 32) row[j] *= inv;
}
// dS = scale * P * (dP - sum(dP * P)) per row, in place on dP
extern "C" __global__ void a_softmax_back(const float* P, float* dP, long long cols, float scale, long long rows) {
    const long long r = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int lane = threadIdx.x & 31;
    const float* p = P + r * cols;
    float* d = dP + r * cols;
    float t = 0.0f;
    for (long long j = lane; j < cols; j += 32) t += d[j] * p[j];
    const float dot = warp_sum(t);
    for (long long j = lane; j < cols; j += 32) d[j] = scale * p[j] * (d[j] - dot);
}
)");
    return s;
}

// Attention straight from one fused projection: qkv [B, S, 3E] holds Q, K and
// V side by side in each row (x @ [W_q W_k W_v], a single gemm), and every
// head is read in place at row stride 3E. With RoPE, one kernel writes the
// rotated Q and K (positions restart at 0 in each sequence) and the scores use
// those; the backward rotates their gradients back in place. The gradient is
// a single [B, S, 3E] tensor laid out like qkv.

static const string& ROPE_SRC() {
    static const string s = src(R"(
// pair i of head h at row (b, t): rotate (v[2i], v[2i+1]) by t * 10000^(-2i/D).
// sign +1 rotates (forward), -1 the inverse (backward). src / dst row strides.
__device__ void rope_pair(float* c, float* s, int t, int i, int D, float sign) {
    const float theta = powf(10000.0f, -2.0f * (float)i / (float)D);
    const float angle = (float)t * theta;
    *c = cosf(angle); *s = sign * sinf(angle);
}
extern "C" __global__ void a_rope(const float* qkv, float* Q, float* K, long long rows, int S, int E, int D) {
    long long n = gid(); const int half = E / 2; if (n >= rows * half) return;
    const long long r = n / half; const int p = (int)(n % half);         // pair p of the row's E dims
    const int h = p / (D / 2), i = p % (D / 2), t = (int)(r % S);
    const long long col = (long long)h * D + 2 * i;
    float c, s; rope_pair(&c, &s, t, i, D, 1.0f);
    const float* src = qkv + r * 3 * E;
    const float q0 = src[col], q1 = src[col + 1], k0 = src[E + col], k1 = src[E + col + 1];
    Q[r * E + col] = q0 * c - q1 * s;  Q[r * E + col + 1] = q0 * s + q1 * c;
    K[r * E + col] = k0 * c - k1 * s;  K[r * E + col + 1] = k0 * s + k1 * c;
}
// pointers for a two-level batch: matrix (b, h) of X starts at
// X + b * sb + h * sh; three operands, B * H matrices, written into ptr[3][B*H]
extern "C" __global__ void a_ptrs(const float* x, long long xb, long long xh, const float* y, long long yb,
                                  long long yh, float* z, long long zb, long long zh, int H, int n,
                                  unsigned long long* ptr) {
    int i = (int)gid(); if (i >= n) return;
    const long long b = i / H, h = i % H;
    ptr[i] = (unsigned long long)(x + b * xb + h * xh);
    ptr[n + i] = (unsigned long long)(y + b * yb + h * yh);
    ptr[2 * n + i] = (unsigned long long)(z + b * zb + h * zh);
}
// the inverse rotation, in place on the Q and K thirds of dqkv [rows, 3E]
extern "C" __global__ void a_rope_back(float* g, long long rows, int S, int E, int D) {
    long long n = gid(); const int half = E / 2; if (n >= rows * half) return;
    const long long r = n / half; const int p = (int)(n % half);
    const int h = p / (D / 2), i = p % (D / 2), t = (int)(r % S);
    const long long col = (long long)h * D + 2 * i;
    float c, s; rope_pair(&c, &s, t, i, D, -1.0f);
    float* row = g + r * 3 * E;
    for (int part = 0; part < 2; part++) {
        float* v = row + part * E + col;
        const float a = v[0], b = v[1];
        v[0] = a * c - b * s; v[1] = a * s + b * c;
    }
}
)");
    return s;
}

// One cuBLAS call for every (sequence, head) pair: matrix (b, h) of each
// operand starts at base + b * stride_b + h * stride_h, a layout no single
// stride describes. Calling per sequence left 4 small products per call --
// too few to fill the GPU, 192 calls a GPT step.
struct Op2 { const float* p; long long sb, sh; int ld; };
static void gemm_bh(bool ta, bool tb, int m, int n, int k, Op2 a, Op2 b, float beta, float* c, long long csb,
                    long long csh, int ldc, int B, int H) {
    const int nb = B * H;
    if (!nb) return;
    // The pointer arrays depend only on the bases and strides, and the caching
    // allocator hands the same buffers back step after step: kept by those
    // values, a training step's 24 pointer-building launches happen once.
    using Key = std::array<long long, 11>;
    struct KeyHash {
        size_t operator()(const Key& k) const {
            size_t h = 1469598103934665603ull;
            for (long long v : k) h = (h ^ (size_t)v) * 1099511628211ull;
            return h;
        }
    };
    // never destroyed: freeing device memory at exit, after the allocator, would crash
    static auto& cache = *new std::unordered_map<Key, TensorPtr, KeyHash>();
    static std::mutex mu;
    const Key key{(long long)(intptr_t)a.p, a.sb, a.sh, (long long)(intptr_t)b.p, b.sb, b.sh,
                  (long long)(intptr_t)c, csb, csh, (long long)H, (long long)nb};
    TensorPtr ptr;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(key);
        if (it != cache.end()) ptr = it->second;
    }
    unsigned long long* pp;
    if (ptr) {
        pp = static_cast<unsigned long long*>(ptr->storage()->data());
    } else {
        ptr = empty_cuda({3LL * nb}, DType::Int64);
        pp = static_cast<unsigned long long*>(ptr->storage()->data());
        const float* x = a.p;
        const float* y = b.p;
        float* z = c;
        long long xb = a.sb, xh = a.sh, yb = b.sb, yh = b.sh, zb = csb, zh = csh;
        int h_ = H, n_ = nb;
        void* args[] = {&x, &xb, &xh, &y, &yb, &yh, &z, &zb, &zh, &h_, &n_, &pp};
        cuda::launch(K("a_ptrs", ROPE_SRC()), nb, args);
        std::lock_guard<std::mutex> lock(mu);
        if (cache.size() >= 1024) cache.clear();       // bounded: a new shape or buffer each step
        cache.emplace(key, ptr);
    }
    auto P = reinterpret_cast<const float* const*>(pp);
    cuda::sgemm_pointers(ta, tb, m, n, k, 1.0f, P, a.ld, P + nb, b.ld, beta,
                         reinterpret_cast<float* const*>(pp + 2 * nb), ldc, nb);
}

TensorPtr attention_qkv(const TensorPtr& qkv0, int heads, bool causal, bool rope) {
    auto qkv = contiguous_on_device(f32(qkv0, "attention"));
    if (qkv->dim() != 3 || qkv->size(2) % 3 != 0)
        throw runtime_error("attention: qkv must be [batch, seq, 3 * embed]");
    const int64_t B = qkv->size(0), S = qkv->size(1), E = qkv->size(2) / 3, H = heads, D = E / H;
    if (D * H != E) throw runtime_error("attention: embed must divide by heads");
    if (rope && D % 2) throw runtime_error("attention: RoPE needs an even head size");
    const float scale = 1.0f / std::sqrt((float)D);
    const int64_t E3 = 3 * E, rows = B * S;
    const float* px = static_cast<const float*>(base(qkv));
    // Q and K as the scores read them: in place in qkv, or rotated copies
    TensorPtr Qr, Kr;
    const float *pq = px, *pk = px + E;
    int64_t lqk = E3;
    if (rope) {
        Qr = empty_cuda({B, S, E});
        Kr = empty_cuda({B, S, E});
        float* q = static_cast<float*>(Qr->storage()->data());
        float* k = static_cast<float*>(Kr->storage()->data());
        long long rr = rows;
        int s_ = (int)S, e_ = (int)E, d_ = (int)D;
        void* a[] = {&px, &q, &k, &rr, &s_, &e_, &d_};
        cuda::launch(K("a_rope", ROPE_SRC()), rows * (E / 2), a);
        pq = q; pk = k; lqk = E;
    }
    const float* pv = px + 2 * E;
    auto P = empty_cuda({B, H, S, S});
    auto O = empty_cuda({B, S, E});
    float* pp = static_cast<float*>(P->storage()->data());
    float* po = static_cast<float*>(O->storage()->data());
    const int s_ = (int)S, d_ = (int)D, e_ = (int)E, h_ = (int)H, lq = (int)lqk, l3 = (int)E3;
    // scores = Q K^T for every head of every sequence, one call
    gemm_bh(false, true, s_, s_, d_, {pq, S * lqk, D, lq}, {pk, S * lqk, D, lq}, 0.0f, pp, H * S * S, S * S, s_,
            (int)B, h_);
    long long cols = S;
    int cz = causal ? 1 : 0;
    float sc = scale;
    long long nrows = B * H * S;
    void* a1[] = {&pp, &cols, &sc, &cz, &nrows};
    if (B * H * S) cuda::launch(K("a_softmax", ATTN_SRC()), (unsigned)((B * H * S + 7) / 8), 1, 1, 256, 1, 1, 0, a1);
    // O = P V, written into the heads' columns
    gemm_bh(false, false, s_, d_, s_, {pp, H * S * S, S * S, s_}, {pv, S * E3, D, l3}, 0.0f, po, S * E, D, e_,
            (int)B, h_);

    if (qkv->requires_grad()) {
        auto node = make_node(O, {qkv});
        node->name = "attention";
        // the op form: what higher-order gradients differentiate
        autograd::attach_reference(*node, [B, S, E, H, D, scale, causal, rope](const vector<TensorPtr>& in) -> TensorPtr {
            auto part = [&](int k) {           // [B, S, E] third k of the rows
                auto idx = make_shared<vector<int64_t>>((size_t)(B * S * E));
                for (int64_t r = 0; r < B * S; r++)
                    for (int64_t j = 0; j < E; j++) (*idx)[(size_t)(r * E + j)] = r * 3 * E + k * E + j;
                return gather_flat(in[0], idx, Shape{B, S, E});
            };
            auto heads_of = [&](const TensorPtr& t) { return permute(reshape(t, {B, S, H, D}), {0, 2, 1, 3}); };
            auto Q = heads_of(part(0)), Kh = heads_of(part(1)), V = heads_of(part(2));
            if (rope) {
                // a constant per-position rotation: v' = v @ R_t
                vector<float> R((size_t)(S * D * D), 0.0f);
                for (int64_t t = 0; t < S; t++)
                    for (int64_t i = 0; i < D / 2; i++) {
                        const float theta = powf(10000.0f, -2.0f * (float)i / (float)D), angle = (float)t * theta;
                        const float c = cosf(angle), sn = sinf(angle);
                        float* r = R.data() + t * D * D;
                        r[(2 * i) * D + 2 * i] = c;       r[(2 * i) * D + 2 * i + 1] = sn;
                        r[(2 * i + 1) * D + 2 * i] = -sn; r[(2 * i + 1) * D + 2 * i + 1] = c;
                    }
                auto Rt = make_shared<Tensor>(Shape{S, D, D}, R)->to(Device::CUDA);
                Q = reshape(::matmul(reshape(Q, {B, H, S, 1, D}), Rt), {B, H, S, D});
                Kh = reshape(::matmul(reshape(Kh, {B, H, S, 1, D}), Rt), {B, H, S, D});
            }
            auto scores = multiply(scale, ::matmul(Q, transpose(Kh)));
            if (causal) {
                vector<float> m((size_t)(S * S), 0.0f);
                for (int64_t i = 0; i < S; i++) for (int64_t j = i + 1; j < S; j++) m[(size_t)(i * S + j)] = -1e9f;
                scores = add(scores, make_shared<Tensor>(Shape{S, S}, m)->to(Device::CUDA));
            }
            auto o = ::matmul(::softmax(scores), V);
            return reshape(permute(o, {0, 2, 1, 3}), {B, S, E});
        });
        auto reference = node->backward_graph;
        node->backward_graph = [reference, qkv, Qr, Kr, P, B, S, E, H, D, scale, rope](
                                   const TensorPtr& out, const TensorPtr& G0) -> vector<TensorPtr> {
            if (grad_enabled()) return reference(out, G0);
            auto G = contiguous_on_device(f32(G0, "attention"));
            const int64_t E3 = 3 * E;
            auto dP = empty_cuda({B, H, S, S});
            auto dX = empty_cuda({B, S, E3});
            const float* px = static_cast<const float*>(base(qkv));
            const float* pq = rope ? static_cast<const float*>(base(Qr)) : px;
            const float* pk = rope ? static_cast<const float*>(base(Kr)) : px + E;
            const int64_t lqk = rope ? E : E3;
            const float* pv = px + 2 * E;
            const float* pp = static_cast<const float*>(base(P));
            const float* pg = static_cast<const float*>(base(G));
            float* pdp = static_cast<float*>(dP->storage()->data());
            float* pdq = static_cast<float*>(dX->storage()->data());
            float* pdk = pdq + E;
            float* pdv = pdq + 2 * E;
            const int s_ = (int)S, d_ = (int)D, e_ = (int)E, h_ = (int)H, lq = (int)lqk, l3 = (int)E3;
            // dV = P^T dO;  dP = dO V^T  -- each one call over every (sequence, head)
            gemm_bh(true, false, s_, d_, s_, {pp, H * S * S, S * S, s_}, {pg, S * E, D, e_}, 0.0f, pdv, S * E3, D,
                    l3, (int)B, h_);
            gemm_bh(false, true, s_, s_, d_, {pg, S * E, D, e_}, {pv, S * E3, D, l3}, 0.0f, pdp, H * S * S, S * S,
                    s_, (int)B, h_);
            long long cols = S;
            float sc = scale;
            long long nrows = B * H * S;
            void* a2[] = {&pp, &pdp, &cols, &sc, &nrows};
            if (B * H * S) cuda::launch(K("a_softmax_back", ATTN_SRC()), (unsigned)((B * H * S + 7) / 8), 1, 1, 256, 1, 1, 0, a2);
            // dQ = dS K;  dK = dS^T Q   (in the rotated space under RoPE)
            gemm_bh(false, false, s_, d_, s_, {pdp, H * S * S, S * S, s_}, {pk, S * lqk, D, lq}, 0.0f, pdq, S * E3,
                    D, l3, (int)B, h_);
            gemm_bh(true, false, s_, d_, s_, {pdp, H * S * S, S * S, s_}, {pq, S * lqk, D, lq}, 0.0f, pdk, S * E3,
                    D, l3, (int)B, h_);
            if (rope) {
                long long rr = B * S;
                int s2 = (int)S, e2 = (int)E, d2 = (int)D;
                void* a3[] = {&pdq, &rr, &s2, &e2, &d2};
                cuda::launch(K("a_rope_back", ROPE_SRC()), B * S * (E / 2), a3);
            }
            return {dX};
        };
    }
    return O;
}

// ── fused transformer pieces: bias + GELU, bias + residual, residual + LayerNorm ──
//
// What a transformer block does between its gemms, each in one kernel instead
// of two or three. Gradients as for the other fused kernels: one kernel for a
// plain backward, the op form when the gradient is itself recorded.

static const string& BLOCK_SRC() {
    static const string s = src(R"(
__device__ float gelu_f(float v) { return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))); }
__device__ float gelu_d(float v) {
    const float k = 0.7978845608f, c = 0.044715f, t = tanhf(k * (v + c * v * v * v));
    return 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * k * (1.0f + 3.0f * c * v * v);
}
// y = gelu(z + b), b broadcast along the last dim
extern "C" __global__ void k_bias_gelu(const float* z, const float* b, float* y, long long n, long long cols) {
    long long i = gid(); if (i >= n) return;
    y[i] = gelu_f(z[i] + b[i % cols]);
}
extern "C" __global__ void k_bias_gelu_back(const float* z, const float* b, const float* dy, float* dz,
                                            long long n, long long cols) {
    long long i = gid(); if (i >= n) return;
    dz[i] = dy[i] * gelu_d(z[i] + b[i % cols]);
}
// out = y + b + r, b broadcast along the last dim
extern "C" __global__ void k_add_bias_residual(const float* y, const float* b, const float* r, float* out,
                                               long long n, long long cols) {
    long long i = gid(); if (i >= n) return;
    out[i] = y[i] + b[i % cols] + r[i];
}
// s = a + c and y = LayerNorm(s), one warp per row with the row in registers
// (rows of up to 1024), as f_layernorm_w
__device__ float alnw_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
extern "C" __global__ void k_add_layernorm_w(const float* a, const float* c, float* s_out, const float* g,
                                             const float* bt, float* y, int cols, float eps, long long rows) {
    const long long r = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int lane = threadIdx.x & 31;
    const long long off = r * cols;
    float v[32];
    float t = 0.0f;
    #pragma unroll
    for (int k = 0; k < 32; k++) {
        const int j = lane + 32 * k;
        v[k] = 0.0f;
        if (j < cols) { v[k] = a[off + j] + c[off + j]; s_out[off + j] = v[k]; }
        t += v[k];
    }
    const float mean = alnw_sum(t) / (float)cols;
    t = 0.0f;
    #pragma unroll
    for (int k = 0; k < 32; k++) { const int j = lane + 32 * k; if (j < cols) { const float d = v[k] - mean; t += d * d; } }
    const float inv = rsqrtf(alnw_sum(t) / (float)cols + eps);
    #pragma unroll
    for (int k = 0; k < 32; k++) { const int j = lane + 32 * k; if (j < cols) y[off + j] = (v[k] - mean) * inv * g[j] + bt[j]; }
}
// s = a + c and y = LayerNorm(s) per row, one block per row (rows over 1024)
extern "C" __global__ void k_add_layernorm(const float* a, const float* c, float* s_out, const float* g,
                                           const float* bt, float* y, long long cols, float eps) {
    __shared__ float sh[256];
    const long long off = (long long)blockIdx.x * cols;
    float t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) { float v = a[off + j] + c[off + j]; s_out[off + j] = v; t += v; }
    sh[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w]; __syncthreads(); }
    const float mean = sh[0] / (float)cols;
    __syncthreads();
    t = 0.0f;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) { float d = s_out[off + j] - mean; t += d * d; }
    sh[threadIdx.x] = t;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) { if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w]; __syncthreads(); }
    const float inv = rsqrtf(sh[0] / (float)cols + eps);
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) y[off + j] = (s_out[off + j] - mean) * inv * g[j] + bt[j];
}
)");
    return s;
}

TensorPtr bias_gelu(const TensorPtr& z0, const TensorPtr& b0) {
    auto z = contiguous_on_device(f32(z0, "bias_gelu")), b = contiguous_on_device(f32(b0, "bias_gelu"));
    long long cols = z->shape().back(), n = z->numel();
    if (b->numel() != cols) throw runtime_error("bias_gelu: bias must have " + to_string(cols) + " elements");
    auto y = empty_cuda(z->shape());
    const void* pz = base(z);
    const void* pb = base(b);
    void* py = y->storage()->data();
    void* args[] = {&pz, &pb, &py, &n, &cols};
    cuda::launch(K("k_bias_gelu", BLOCK_SRC()), n, args);
    if (z->requires_grad() || b->requires_grad()) {
        auto node = make_node(y, {z, b});
        node->name = "bias_gelu";
        autograd::attach_reference(*node, [](const vector<TensorPtr>& in) -> TensorPtr { return gelu(add(in[0], in[1])); });
        auto reference = node->backward_graph;
        node->backward_graph = [reference, z, b, n, cols](const TensorPtr& out, const TensorPtr& G0) -> vector<TensorPtr> {
            if (grad_enabled()) return reference(out, G0);
            auto G = contiguous_on_device(f32(G0, "bias_gelu"));
            auto dz = empty_cuda(z->shape());
            const void* pz = base(z);
            const void* pb = base(b);
            const void* pg = base(G);
            void* pd = dz->storage()->data();
            long long nn = n, cc = cols;
            void* a[] = {&pz, &pb, &pg, &pd, &nn, &cc};
            cuda::launch(K("k_bias_gelu_back", BLOCK_SRC()), nn, a);
            TensorPtr db = b->requires_grad() ? ::sum(reshape(dz, {n / cols, cols}), 0) : nullptr;
            return {z->requires_grad() ? dz : nullptr, db};
        };
    }
    return y;
}

TensorPtr add_bias_residual(const TensorPtr& y0, const TensorPtr& b0, const TensorPtr& r0) {
    auto y = contiguous_on_device(f32(y0, "add_bias_residual")), b = contiguous_on_device(f32(b0, "add_bias_residual"));
    auto r = contiguous_on_device(f32(r0, "add_bias_residual"));
    long long cols = y->shape().back(), n = y->numel();
    if (b->numel() != cols || r->shape() != y->shape())
        throw runtime_error("add_bias_residual: bias must be [cols] and the residual the shape of y");
    auto out = empty_cuda(y->shape());
    const void* py = base(y);
    const void* pb = base(b);
    const void* pr = base(r);
    void* po = out->storage()->data();
    void* args[] = {&py, &pb, &pr, &po, &n, &cols};
    cuda::launch(K("k_add_bias_residual", BLOCK_SRC()), n, args);
    attach(out, {y, b, r}, "add_bias_residual", [y, b, r](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        return {y->requires_grad() ? G : nullptr, b->requires_grad() ? sum_to(G, b->shape()) : nullptr,
                r->requires_grad() ? G : nullptr};
    });
    return out;
}

pair<TensorPtr, TensorPtr> add_layernorm(const TensorPtr& a0, const TensorPtr& c0, const TensorPtr& g0,
                                         const TensorPtr& bt0, float eps) {
    auto a = contiguous_on_device(f32(a0, "add_layernorm")), c = contiguous_on_device(f32(c0, "add_layernorm"));
    auto g = contiguous_on_device(f32(g0, "add_layernorm")), bt = contiguous_on_device(f32(bt0, "add_layernorm"));
    if (a->shape() != c->shape()) throw runtime_error("add_layernorm: the two inputs differ in shape");
    long long cols = a->shape().back();
    const long long rows = cols ? a->numel() / cols : 0;
    auto s = empty_cuda(a->shape()), y = empty_cuda(a->shape());
    const void* pa = base(a);
    const void* pc = base(c);
    const void* pg = base(g);
    const void* pbt = base(bt);
    void* ps = s->storage()->data();
    void* py = y->storage()->data();
    if (rows && cols <= 1024) {
        int cc = (int)cols;
        long long rr = rows;
        void* a1[] = {&pa, &pc, &ps, &pg, &pbt, &py, &cc, &eps, &rr};
        cuda::launch(K("k_add_layernorm_w", BLOCK_SRC()), (unsigned)((rows + 7) / 8), 1, 1, 256, 1, 1, 0, a1);
    } else {
        void* args[] = {&pa, &pc, &ps, &pg, &pbt, &py, &cols, &eps};
        if (rows) cuda::launch(K("k_add_layernorm", BLOCK_SRC()), (unsigned)rows, 1, 1, 256, 1, 1, 0, args);
    }
    // s is a + c; y is LayerNorm(s), with the LayerNorm gradient: fused, or the op form
    attach(s, {a, c}, "add", [a, c](const TensorPtr&, const TensorPtr& G) -> vector<TensorPtr> {
        return {a->requires_grad() ? G : nullptr, c->requires_grad() ? G : nullptr};
    });
    if (s->requires_grad() || g->requires_grad() || bt->requires_grad()) {
        auto node = make_node(y, {s, g, bt});
        node->name = "LayerNorm::forward";
        autograd::attach_reference(*node, [eps](const vector<TensorPtr>& in) -> TensorPtr {
            const int last = in[0]->dim() - 1;
            auto m = mean(in[0], last, true);
            auto d = subtract(in[0], m);
            auto var = mean(multiply(d, d), last, true);
            return add(multiply(divide(d, sqrt_op(add(eps, var))), in[1]), in[2]);
        });
        auto reference = node->backward_graph;
        node->backward_graph = [reference, s, g, bt, eps](const TensorPtr& out, const TensorPtr& G) -> vector<TensorPtr> {
            if (grad_enabled()) return reference(out, G);
            auto r = layernorm_backward(s, g, G, eps,
                                        backward_into_grads() && s->has_grad() ? s->grad_tensor() : nullptr);
            return {s->requires_grad() ? r[0] : nullptr, g->requires_grad() ? r[1] : nullptr,
                    bt->requires_grad() ? r[2] : nullptr};
        };
    }
    return {s, y};
}

} // namespace cuda_ops
