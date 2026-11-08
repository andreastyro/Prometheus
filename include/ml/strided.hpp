#pragma once
#include <cstdint>
#include <vector>
#include "ml/tensor.hpp"

namespace strided {

/// Call f(i, offset) for every element of a strided layout, i running in
/// row-major order of `shape` and offset being that element's index into
/// storage. The offset is advanced incrementally, so there is no
/// multiply-per-dimension per element.
template <typename F>
inline void for_each(const Shape& shape, const Shape& strides,
                     std::int64_t offset, F&& f) {
    std::int64_t n = 1;
    for (std::int64_t s : shape) n *= s;
    if (n == 0) return;

    const int nd = (int)shape.size();
    std::vector<std::int64_t> idx(nd, 0);
    std::int64_t off = offset;
    for (std::int64_t i = 0; i < n; i++) {
        f(i, off);
        for (int d = nd - 1; d >= 0; d--) {
            if (++idx[d] < shape[d]) { off += strides[d]; break; }
            off -= strides[d] * (shape[d] - 1);
            idx[d] = 0;
        }
    }
}

/// Element `index` of a buffer of `dtype`, as a double. Conversions go
/// through double rather than float so that int64 values are exact up to 2^53
/// -- through float, token ids above 2^24 would round.
double load(const void* base, DType dtype, std::int64_t index);
/// Store `v` as element `index` of a buffer of `dtype`, rounding to nearest
/// for the narrow float formats.
void   store(void* base, DType dtype, std::int64_t index, double v);

} // namespace strided
