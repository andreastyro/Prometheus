#include "ml/tensor.hpp"
#include "ml/utils/grad_clip.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    // Two parameters, each with a gradient of 3.0 and 4.0 → global norm = 5.0
    auto p1 = std::make_shared<Tensor>(std::vector<int>{1, 1}, std::vector<float>{0.0f});
    auto p2 = std::make_shared<Tensor>(std::vector<int>{1, 1}, std::vector<float>{0.0f});
    p1->grad = {3.0f};
    p2->grad = {4.0f};

    std::vector<TensorPtr> params = {p1, p2};

    // Clip to max_norm = 2.5 (well below 5.0)
    float norm = clip_grad_norm(params, 2.5f);
    assert(std::fabs(norm - 5.0f) < 1e-4f && "reported norm should be pre-clip value");

    // After clipping: scale = 2.5 / 5.0 = 0.5
    assert(std::fabs(p1->grad[0] - 1.5f) < 1e-4f);
    assert(std::fabs(p2->grad[0] - 2.0f) < 1e-4f);

    // If norm is already below max_norm, gradients should be unchanged
    auto p3 = std::make_shared<Tensor>(std::vector<int>{1, 1}, std::vector<float>{0.0f});
    p3->grad = {0.1f};
    std::vector<TensorPtr> small = {p3};
    float norm2 = clip_grad_norm(small, 10.0f);
    assert(std::fabs(norm2 - 0.1f) < 1e-4f);
    assert(std::fabs(p3->grad[0] - 0.1f) < 1e-5f && "no-op when norm < max_norm");

    printf("test_grad_clip PASSED\n");
    return 0;
}
