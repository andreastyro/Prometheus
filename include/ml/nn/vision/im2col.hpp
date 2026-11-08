#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
#include "ml/matmul_backend.hpp"
#ifdef PROMETHEUS_USE_OPENMP
    #include <omp.h>
#endif

/// im2col / col2im: the step that turns a convolution into a matrix multiply,
/// which is how PyTorch runs convolution on a CPU without oneDNN.
///
/// For an image [C, H, W] and a K x K kernel with stride s and padding p, the
/// column matrix has one row per (channel, kernel row, kernel column) and one
/// column per output position:
///
///     col[(c, kh, kw), (i, j)] = img[c, i*s + kh - p, j*s + kw - p]   (0 outside)
///
/// so a convolution of one image is  out[F, Ho*Wo] = W[F, C*K*K] @ col.
/// col2im is the adjoint: it adds each column entry back onto the pixel it
/// came from. A transposed convolution's forward pass is exactly that, and so
/// is a convolution's input gradient.
namespace vision {

/// Already inside a parallel region? Then the loops below run serially rather
/// than relying on the runtime's nesting default, which differs between
/// OpenMP implementations.
inline bool omp_in_parallel_flag() {
#ifdef PROMETHEUS_USE_OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

/// img: one image [C, H, W]; col: [C*K*K, Ho*Wo], fully overwritten.
inline void im2col(const float* img, int C, int H, int W,
                   int K, int s, int p, int Ho, int Wo, float* col) {
    const int64_t HW = (int64_t)Ho * Wo;
    // each channel writes its own K*K rows; parallel only when called on its
    // own -- inside a per-image parallel loop this runs on one thread
    #pragma omp parallel for schedule(static) if(!omp_in_parallel_flag() && (int64_t)C * K * K * HW >= 65536)
    for (int c = 0; c < C; c++) {
        const float* src_c = img + (int64_t)c * H * W;
        for (int kh = 0; kh < K; kh++)
            for (int kw = 0; kw < K; kw++) {
                float* row = col + ((int64_t)(c * K + kh) * K + kw) * HW;
                for (int i = 0; i < Ho; i++) {
                    float* dst = row + (int64_t)i * Wo;
                    const int y = i * s + kh - p;
                    if (y < 0 || y >= H) { std::fill(dst, dst + Wo, 0.0f); continue; }
                    const float* src = src_c + (int64_t)y * W;
                    // the interior of the row needs no bounds check
                    const int x0 = kw - p;
                    int jlo = 0, jhi = Wo;
                    while (jlo < Wo && jlo * s + x0 < 0) dst[jlo++] = 0.0f;
                    while (jhi > jlo && (jhi - 1) * s + x0 >= W) dst[--jhi] = 0.0f;
                    if (s == 1) std::copy(src + jlo + x0, src + jhi + x0, dst + jlo);
                    else for (int j = jlo; j < jhi; j++) dst[j] = src[j * s + x0];
                }
            }
    }
}

/// The adjoint: adds col [C*K*K, Ho*Wo] onto one image [C, H, W].
inline void col2im_add(const float* col, int C, int H, int W,
                       int K, int s, int p, int Ho, int Wo, float* img) {
    const int64_t HW = (int64_t)Ho * Wo;
    // each channel writes only its own plane
    #pragma omp parallel for schedule(static) if(!omp_in_parallel_flag() && (int64_t)C * K * K * HW >= 65536)
    for (int c = 0; c < C; c++) {
        float* dst_c = img + (int64_t)c * H * W;
        for (int kh = 0; kh < K; kh++)
            for (int kw = 0; kw < K; kw++) {
                const float* row = col + ((int64_t)(c * K + kh) * K + kw) * HW;
                for (int i = 0; i < Ho; i++) {
                    const int y = i * s + kh - p;
                    if (y < 0 || y >= H) continue;
                    const float* src = row + (int64_t)i * Wo;
                    float* dst = dst_c + (int64_t)y * W;
                    const int x0 = kw - p;
                    int jlo = 0, jhi = Wo;
                    while (jlo < Wo && jlo * s + x0 < 0) jlo++;
                    while (jhi > jlo && (jhi - 1) * s + x0 >= W) jhi--;
                    if (s == 1) for (int j = jlo; j < jhi; j++) dst[j + x0] += src[j];
                    else        for (int j = jlo; j < jhi; j++) dst[j * s + x0] += src[j];
                }
            }
    }
}

inline int thread_id() {
#ifdef PROMETHEUS_USE_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

/// Run body(b, slot) once per image of the batch.
///
/// With enough images to occupy the cores, images run in parallel, each with
/// a single-threaded BLAS gemm: per-image work stays in cache, there is no
/// batch transpose, and nothing contends. With few images (inference, B = 1)
/// they run one at a time and BLAS uses every core instead. `slot` indexes
/// per-thread scratch; slots() says how many a caller needs.
struct ImageLoop {
    int  B;
    bool parallel;
    explicit ImageLoop(int batch)
        : B(batch), parallel(batch >= std::max(2, matmul_max_threads() / 2)) {}
    int slots() const { return parallel ? matmul_max_threads() : 1; }

    template <class Fn> void run(Fn body) const {
        if (parallel) {
            SerialBlasScope serial;
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < B; b++) body(b, thread_id());
        } else {
            for (int b = 0; b < B; b++) body(b, 0);
        }
    }
};

/// Sum per-thread partial buffers into dst.
inline void reduce_partials(std::vector<std::vector<float>>& parts, float* dst, int64_t n) {
    for (auto& part : parts)
        if (!part.empty())
            for (int64_t i = 0; i < n; i++) dst[i] += part[i];
}

} // namespace vision
