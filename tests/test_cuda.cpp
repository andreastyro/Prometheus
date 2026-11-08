// GPU tensors against the CPU. Skips (passes) on a machine without a GPU.
//
// Point PROMETHEUS_CUDA_PATH at the folder holding NVRTC (nvrtc64_120_0.dll /
// libnvrtc.so.12) if it is not on the library path.
#include "ml/cuda.hpp"
#include "ml/ops.hpp"
#include "ml/tensor.hpp"
#include "ml/utils/testing.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static TensorPtr random_tensor(Shape shape, unsigned seed) {
    auto t = std::make_shared<Tensor>(shape);
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(0.5f, 2.0f);
    for (float& v : t->values()) v = u(g);
    return t;
}

static float max_diff(const TensorPtr& gpu, const TensorPtr& cpu) {
    const auto a = gpu->to_vector(), b = cpu->to_vector();
    if (a.size() != b.size()) return INFINITY;
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    if (const char* dir = std::getenv("PROMETHEUS_CUDA_PATH")) cuda::set_search_dirs({dir});
    std::printf("cuda: %s\n", cuda::status().c_str());
    if (!cuda::available()) {
        std::printf("no GPU: skipped\n");
        return 0;
    }

    // copies there and back are exact, for every dtype and for views
    auto x = random_tensor({3, 5}, 1);
    auto xg = x->to(Device::CUDA);
    CHECK_TRUE("tensor lands on the GPU", xg->device() == Device::CUDA);
    CHECK_TRUE("round trip is exact", max_diff(xg, x) == 0.0f);
    auto ids = std::make_shared<Tensor>(Shape{4}, DType::Int64);
    ids->set_data({1, 2, 3, 4});
    CHECK_TRUE("int64 round trip", max_diff(ids->to(Device::CUDA), ids) == 0.0f);
    auto xt = transpose(xg);    // a strided view of GPU memory
    CHECK_TRUE("a transposed GPU view reads back transposed",
               max_diff(xt, transpose(x)) == 0.0f);
    auto one = std::make_shared<Tensor>(Shape{1}, std::vector<float>{3.5f})->to(Device::CUDA);
    CHECK_NEAR("item() reads from the GPU", (float)one->item_double(), 3.5f, 0.0f);

    // element-wise ops, including broadcasting and a view operand
    auto y = random_tensor({3, 5}, 2);
    auto row = random_tensor({5}, 3);
    auto yg = y->to(Device::CUDA), rowg = row->to(Device::CUDA);
    CHECK_TRUE("add",      max_diff(add(xg, yg), add(x, y)) == 0.0f);
    CHECK_TRUE("subtract", max_diff(subtract(xg, yg), subtract(x, y)) == 0.0f);
    CHECK_TRUE("multiply", max_diff(multiply(xg, yg), multiply(x, y)) == 0.0f);
    CHECK_NEAR("divide",   max_diff(divide(xg, yg), divide(x, y)), 0.0f, 1e-6f);
    CHECK_TRUE("broadcast a row", max_diff(add(xg, rowg), add(x, row)) == 0.0f);
    auto sq = random_tensor({5, 5}, 4);
    auto sqg = sq->to(Device::CUDA);
    CHECK_TRUE("view operand (transpose)",
               max_diff(multiply(transpose(sqg), sqg), multiply(transpose(sq), sq)) == 0.0f);

    // a large one, to cover many blocks
    auto big = random_tensor({1000, 1000}, 5), big2 = random_tensor({1000, 1000}, 6);
    CHECK_TRUE("1M elements", max_diff(add(big->to(Device::CUDA), big2->to(Device::CUDA)),
                                       add(big, big2)) == 0.0f);

    bool mixed = false;
    try { add(xg, y); } catch (const std::exception&) { mixed = true; }
    CHECK_TRUE("mixing devices raises", mixed);
    return test_report();
}
