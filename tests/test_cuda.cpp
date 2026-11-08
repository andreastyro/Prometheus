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
#include <functional>
#include <string>
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

    // ── element-wise math: GPU expf / tanhf differ from the CPU's by an ulp ──
    const float tol = 2e-6f;
    auto G = [](const TensorPtr& t) { return t->to(Device::CUDA); };
    CHECK_NEAR("scalar add",      max_diff(add(2.5f, xg), add(2.5f, x)), 0.0f, tol);
    CHECK_NEAR("scalar multiply", max_diff(multiply(-1.5f, xg), multiply(-1.5f, x)), 0.0f, tol);
    CHECK_NEAR("scalar - x",      max_diff(subtract(1.0f, xg), subtract(1.0f, x)), 0.0f, tol);
    CHECK_NEAR("x / scalar",      max_diff(divide(xg, 3.0f), divide(x, 3.0f)), 0.0f, tol);
    CHECK_NEAR("exp",     max_diff(exp_op(xg), exp_op(x)), 0.0f, 1e-5f);
    CHECK_NEAR("log",     max_diff(log_op(xg), log_op(x)), 0.0f, tol);
    CHECK_NEAR("sqrt",    max_diff(sqrt_op(xg), sqrt_op(x)), 0.0f, tol);
    CHECK_NEAR("pow 1.7", max_diff(pow_op(xg, 1.7f), pow_op(x, 1.7f)), 0.0f, 1e-5f);
    auto z = subtract(x, 1.2f), zg = G(z);     // both signs
    CHECK_NEAR("abs",     max_diff(abs_op(zg), abs_op(z)), 0.0f, 0.0f);
    CHECK_NEAR("relu",    max_diff(relu(zg), relu(z)), 0.0f, 0.0f);
    CHECK_NEAR("sigmoid", max_diff(sigmoid(zg), sigmoid(z)), 0.0f, tol);
    CHECK_NEAR("tanh",    max_diff(tanh_op(zg), tanh_op(z)), 0.0f, tol);
    CHECK_NEAR("gelu",    max_diff(gelu(zg), gelu(z)), 0.0f, 1e-5f);
    CHECK_NEAR("clip",    max_diff(clip(zg, -0.3f, 0.4f), clip(z, -0.3f, 0.4f)), 0.0f, 0.0f);

    // ── reductions, softmax, layout ──
    auto r3 = random_tensor({4, 6, 5}, 7), r3g = G(r3);
    CHECK_NEAR("sum over dim 1",   max_diff(sum(r3g, 1), sum(r3, 1)), 0.0f, 1e-5f);
    CHECK_NEAR("mean over dim 2, keepdim", max_diff(mean(r3g, 2, true), mean(r3, 2, true)), 0.0f, 1e-6f);
    CHECK_NEAR("sum of everything", max_diff(sum(r3g, -1), sum(r3, -1)), 0.0f, 1e-3f);
    CHECK_NEAR("sum of 1M (two-pass)", max_diff(sum(G(big), -1), sum(big, -1)), 0.0f, 2.0f);
    CHECK_NEAR("softmax", max_diff(softmax(r3g), softmax(r3)), 0.0f, 1e-6f);
    auto wide = random_tensor({3, 1000}, 8);
    CHECK_NEAR("softmax, 1000 columns", max_diff(softmax(G(wide)), softmax(wide)), 0.0f, 1e-7f);
    CHECK_TRUE("contiguous of a transposed view",
               max_diff(contiguous(transpose(r3g, 0, 2)), contiguous(transpose(r3, 0, 2))) == 0.0f);

    // ── comparisons, masks, casts ──
    CHECK_TRUE("gt gives bool", gt(zg, 0.0)->dtype() == DType::Bool);
    CHECK_TRUE("gt matches CPU", max_diff(gt(zg, 0.0), gt(z, 0.0)) == 0.0f);
    CHECK_TRUE("logical_and", max_diff(logical_and(gt(zg, -0.2), lt(zg, 0.3)),
                                       logical_and(gt(z, -0.2), lt(z, 0.3))) == 0.0f);
    CHECK_TRUE("where", max_diff(where(gt(zg, 0.0), xg, yg), where(gt(z, 0.0), x, y)) == 0.0f);
    CHECK_TRUE("masked_fill", max_diff(masked_fill(xg, gt(zg, 0.0), -1e9),
                                       masked_fill(x, gt(z, 0.0), -1e9)) == 0.0f);
    for (DType d : {DType::Float16, DType::BFloat16, DType::Int32, DType::Int64, DType::Float64}) {
        CHECK_TRUE((std::string("cast to ") + dtype_name(d) + " rounds as on the CPU").c_str(),
                   max_diff(cast_op(zg, d), cast_op(z, d)) == 0.0f);
    }

    // ── gradients, computed on the GPU, against the CPU's ──
    auto grad_pair = [&](const char* what, const std::function<TensorPtr(TensorPtr, TensorPtr)>& f,
                         float gtol) {
        auto a = random_tensor({3, 5}, 11), b = random_tensor({5}, 12);
        auto ag = G(a), bg = G(b);
        for (auto* t : {&a, &b, &ag, &bg}) (*t)->set_requires_grad(true);
        sum(f(a, b), -1)->backward();
        sum(f(ag, bg), -1)->backward();
        const auto ga = ag->grad_doubles(), gb = bg->grad_doubles();
        const auto ca = a->grad_doubles(), cb = b->grad_doubles();
        double m = 0;
        for (size_t i = 0; i < ca.size(); i++) m = std::max(m, std::fabs(ga[i] - ca[i]));
        for (size_t i = 0; i < cb.size(); i++) m = std::max(m, std::fabs(gb[i] - cb[i]));
        CHECK_NEAR((std::string("gradient: ") + what).c_str(), (float)m, 0.0f, gtol);
    };
    grad_pair("a*b + a/b (broadcast)", [](TensorPtr a, TensorPtr b) { return add(multiply(a, b), divide(a, b)); }, 1e-5f);
    grad_pair("tanh(a) * sigmoid(b)", [](TensorPtr a, TensorPtr b) { return multiply(tanh_op(a), sigmoid(b)); }, 1e-5f);
    grad_pair("softmax(a*b)", [](TensorPtr a, TensorPtr b) {
        return multiply(softmax(multiply(a, b)), add(1.0f, a)); }, 1e-5f);
    grad_pair("gelu, exp, log, sqrt", [](TensorPtr a, TensorPtr b) {
        return add(gelu(subtract(a, b)), multiply(exp_op(a), multiply(log_op(b), sqrt_op(a)))); }, 1e-4f);
    grad_pair("relu, abs, clip", [](TensorPtr a, TensorPtr b) {
        auto d = subtract(a, b);
        return add(relu(d), add(abs_op(d), clip(d, -0.3f, 0.3f))); }, 1e-6f);
    grad_pair("mean over a dim, where", [](TensorPtr a, TensorPtr b) {
        auto m = mean(multiply(a, b), 1, true);
        return where(gt(a, 1.0), multiply(a, m), b); }, 1e-5f);

    // a CPU leaf that goes to the GPU and back gets its gradient on the CPU
    auto leaf = random_tensor({4}, 13);
    leaf->set_requires_grad(true);
    sum(multiply(3.0f, leaf->to(Device::CUDA)), -1)->to(Device::CPU)->backward();
    const auto lg = leaf->grad_doubles();
    CHECK_TRUE("gradient crosses devices (to(cuda) / to(cpu))",
               lg.size() == 4 && std::fabs(lg[0] - 3.0) < 1e-6 && std::fabs(lg[3] - 3.0) < 1e-6);

    // second derivative on the GPU: d2/dx2 tanh(x) = -2 tanh(x) (1 - tanh^2 x)
    auto w = G(std::make_shared<Tensor>(Shape{2}, std::vector<float>{0.7f, -1.3f}));
    w->set_requires_grad(true);
    auto g1 = autograd::grad(sum(tanh_op(w), -1), {w}, true)[0];
    auto g2 = autograd::grad(sum(g1, -1), {w})[0]->to(Device::CPU)->to_vector();
    const float t0 = std::tanh(0.7f), t1 = std::tanh(-1.3f);
    CHECK_NEAR("second derivative on the GPU", std::max(std::fabs(g2[0] + 2 * t0 * (1 - t0 * t0)),
                                                         std::fabs(g2[1] + 2 * t1 * (1 - t1 * t1))), 0.0f, 1e-6f);
    return test_report();
}
