#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>

/// Branch-free exp and tanh that the compiler can vectorise.
///
/// libm's expf and tanhf are opaque calls, so a loop over them runs one
/// element at a time. These are plain arithmetic -- a Cody-Waite range
/// reduction, a degree-6 polynomial and an exponent assembled from bits -- so
/// the same loop runs 8 lanes wide under AVX2. Relative error against libm is
/// about 1e-7, a couple of ulp.
///
/// Inputs are clamped to [-87, 88]: below that e^x is under FLT_MIN and is
/// returned as ~1e-38 rather than 0, which no softmax or GELU can tell apart.
namespace fastmath {

inline float exp(float x) {
    x = std::min(88.0f, std::max(-87.0f, x));

    // n = round(x / ln2), using the 1.5 * 2^23 trick so there is no call to
    // roundf to block vectorisation
    const float shifter = 12582912.0f;
    const float n = (x * 1.44269504088896341f + shifter) - shifter;

    // r = x - n*ln2, with ln2 split in two so the subtraction stays exact
    float r = x - n * 0.693359375f;
    r = r + n * 2.12194440e-4f;

    // e^r for |r| <= ln2/2
    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;

    // 2^n straight into the exponent bits
    const int32_t bits = ((int32_t)n + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof scale);
    return p * scale;
}

/// tanh(x) = 1 - 2 / (e^{2x} + 1). exp clamps, so this saturates cleanly to
/// +-1 for large |x| instead of producing inf/inf.
inline float tanh(float x) {
    return 1.0f - 2.0f / (exp(2.0f * x) + 1.0f);
}

} // namespace fastmath
