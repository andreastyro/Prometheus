// GPU tensors against the CPU. Skips (passes) on a machine without a GPU.
//
// Point PROMETHEUS_CUDA_PATH at the folder holding NVRTC (nvrtc64_120_0.dll /
// libnvrtc.so.12) if it is not on the library path.
#include "ml/cuda.hpp"
#include "ml/ops.hpp"
#include "ml/tensor.hpp"
#include "ml/utils/testing.hpp"
#include "ml/loss.hpp"
#include "ml/optim/grad_scaler.hpp"
#include "ml/optim/adam.hpp"
#include "ml/optim/adamw.hpp"
#include "ml/optim/rmsprop.hpp"
#include "ml/optim/sgd.hpp"
#include "ml/utils/grad_clip.hpp"
#include <cstdio>
#include "ml/nn/linear.hpp"
#include "ml/nn/dropout.hpp"
#include "ml/nn/batchnorm.hpp"
#include "ml/nn/flatten.hpp"
#include "ml/nn/groupnorm.hpp"
#include "ml/nn/rnn/gru.hpp"
#include "ml/nn/rnn/lstm.hpp"
#include "ml/nn/rnn/rnn.hpp"
#include "ml/nn/vision/avgpool2d.hpp"
#include "ml/nn/vision/conv2d.hpp"
#include "ml/nn/vision/convtranspose2d.hpp"
#include "ml/nn/vision/maxpool2d.hpp"
#include "ml/nn/nlp/attention.hpp"
#include "ml/nn/nlp/embedding.hpp"
#include "ml/nn/nlp/gpt.hpp"
#include "ml/nn/nlp/layernorm.hpp"
#include "ml/nn/nlp/positional_encoding.hpp"
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

    // ── indexing: extremes, argmax, gather / scatter, concat ──
    CHECK_TRUE("max_op",  max_diff(max_op(zg), max_op(z)) == 0.0f);
    CHECK_TRUE("min_op",  max_diff(min_op(zg), min_op(z)) == 0.0f);
    CHECK_TRUE("max_op over 1M", max_diff(max_op(G(big)), max_op(big)) == 0.0f);
    CHECK_TRUE("argmax per row", max_diff(argmax(r3g), argmax(r3)) == 0.0f);
    auto idx = std::make_shared<std::vector<int64_t>>(std::vector<int64_t>{4, 0, -1, 14, 4, 7});
    CHECK_TRUE("gather_flat (repeat, padding)",
               max_diff(gather_flat(xg, idx, Shape{2, 3}), gather_flat(x, idx, Shape{2, 3})) == 0.0f);
    auto vals = random_tensor({6}, 9);
    CHECK_NEAR("scatter_add_flat (repeated index adds)",
               max_diff(scatter_add_flat(G(vals), idx, Shape{15}), scatter_add_flat(vals, idx, Shape{15})), 0.0f, 1e-6f);
    CHECK_TRUE("concat axis 0", max_diff(concat_op({xg, yg}, 0), concat_op({x, y}, 0)) == 0.0f);
    CHECK_TRUE("concat axis 1", max_diff(concat_op({xg, yg, xg}, 1), concat_op({x, y, x}, 1)) == 0.0f);

    // ── matmul on cuBLAS: summation order differs from the CPU, so relative ──
    auto rel = [&](const TensorPtr& gpu, const TensorPtr& cpu) {
        const auto a = gpu->to_vector(), b = cpu->to_vector();
        float m = 0.0f, s = 0.0f;
        for (size_t i = 0; i < b.size(); i++) { m = std::max(m, std::fabs(a[i] - b[i])); s = std::max(s, std::fabs(b[i])); }
        return a.size() == b.size() ? m / std::max(s, 1e-12f) : INFINITY;
    };
    std::printf("cuBLAS: %s\n", cuda::blas_status().c_str());
    auto A = random_tensor({64, 96}, 21), B = random_tensor({96, 48}, 22);
    CHECK_NEAR("matmul 64x96 @ 96x48", rel(matmul(G(A), G(B)), matmul(A, B)), 0.0f, 1e-6f);
    auto A3 = random_tensor({3, 17, 33}, 23), B3 = random_tensor({3, 33, 9}, 24);
    CHECK_NEAR("batched matmul (strided batched)", rel(matmul(G(A3), G(B3)), matmul(A3, B3)), 0.0f, 1e-6f);
    auto B1 = random_tensor({1, 33, 9}, 25);
    CHECK_NEAR("batched matmul, broadcast batch", rel(matmul(G(A3), G(B1)), matmul(A3, B1)), 0.0f, 1e-6f);
    auto W = random_tensor({33, 9}, 26);
    CHECK_NEAR("N-D @ weight (one gemm)", rel(matmul(G(A3), G(W)), matmul(A3, W)), 0.0f, 1e-6f);
    CHECK_NEAR("matmul with a transposed view", rel(matmul(G(B), transpose(G(B))), matmul(B, transpose(B))), 0.0f, 1e-6f);
    auto Wt = random_tensor({9, 33}, 27);
    CHECK_NEAR("matmul_bt", rel(matmul_bt(G(random_tensor({5, 33}, 28)), G(Wt)),
                                matmul_bt(random_tensor({5, 33}, 28), Wt)), 0.0f, 1e-6f);
    auto bias = random_tensor({9}, 29);
    CHECK_NEAR("linear_op x @ W + b", rel(linear_op(G(A3), G(W), G(bias)), linear_op(A3, W, bias)), 0.0f, 1e-6f);
    auto L1 = random_tensor({512, 512}, 30), L2 = random_tensor({512, 512}, 31);
    CHECK_NEAR("matmul 512x512x512", rel(matmul(G(L1), G(L2)), matmul(L1, L2)), 0.0f, 1e-5f);

    // gradients through matmul, linear, gather and concat
    {
        auto a = random_tensor({2, 4, 6}, 41), w = random_tensor({6, 3}, 42), b = random_tensor({3}, 43);
        auto ag = G(a), wg = G(w), bg = G(b);
        for (auto* t : {&a, &w, &b, &ag, &wg, &bg}) (*t)->set_requires_grad(true);
        auto f = [&](TensorPtr x, TensorPtr W_, TensorPtr b_) {
            auto h = tanh_op(linear_op(x, W_, b_));
            auto cat = concat_op({h, multiply(2.0f, h)}, 2);
            auto picked = gather_flat(cat, idx, Shape{6});
            return add(sum(matmul(h, transpose(W_)), -1), sum(picked, -1));
        };
        f(a, w, b)->backward();
        f(ag, wg, bg)->backward();
        double m = 0;
        for (auto pr : {std::make_pair(ag, a), std::make_pair(wg, w), std::make_pair(bg, b)}) {
            const auto g1 = pr.first->grad_doubles(), g2 = pr.second->grad_doubles();
            for (size_t i = 0; i < g2.size(); i++) m = std::max(m, std::fabs(g1[i] - g2[i]) / std::max(1.0, std::fabs(g2[i])));
        }
        CHECK_NEAR("gradients through linear, matmul, concat, gather", (float)m, 0.0f, 1e-5f);
    }

    // ── layers: the same module, once on each device ──
    {
        // compare a layer's output and parameter gradients across devices
        auto layer_check = [&](const char* what, auto make, const TensorPtr& input, float tol) {
            Tensor::manual_seed(5);
            auto cpu = make();
            Tensor::manual_seed(5);
            auto gpu = make();
            gpu->to(Device::CUDA);
            auto xc = input->detach(), xg = input->to(Device::CUDA);
            if (is_floating(input->dtype())) { xc->set_requires_grad(true); xg->set_requires_grad(true); }
            Tensor::manual_seed(9);
            auto yc = cpu->forward(xc);
            Tensor::manual_seed(9);
            auto yg = gpu->forward(xg);
            CHECK_TRUE((std::string(what) + ": output on the GPU").c_str(), yg->device() == Device::CUDA);
            CHECK_NEAR((std::string(what) + ": output").c_str(), rel(yg, yc), 0.0f, tol);
            auto wts = random_tensor(yc->shape(), 77);
            sum(multiply(yc, wts), -1)->backward();
            sum(multiply(yg, G(wts)), -1)->backward();
            double m = 0, s = 0;
            auto pc = cpu->parameters(), pg = gpu->parameters();
            if (is_floating(input->dtype())) { pc.push_back(xc); pg.push_back(xg); }
            for (size_t i = 0; i < pc.size(); i++) {
                const auto a = pg[i]->grad_doubles(), b = pc[i]->grad_doubles();
                for (size_t j = 0; j < b.size(); j++) { m = std::max(m, std::fabs(a[j] - b[j])); s = std::max(s, std::fabs(b[j])); }
            }
            CHECK_NEAR((std::string(what) + ": gradients").c_str(), (float)(m / std::max(s, 1e-12)), 0.0f, tol);
        };
        auto x3 = random_tensor({2, 7, 16}, 50);
        layer_check("LayerNorm", [] { return std::make_shared<LayerNorm>(16); }, x3, 1e-5f);
        layer_check("Linear", [] { return std::make_shared<Linear>(16, 8); }, x3, 1e-5f);
        layer_check("MultiHeadAttention", [] { return std::make_shared<MultiHeadAttention>(16, 4); }, x3, 1e-5f);
        layer_check("causal attention with RoPE", [] { return std::make_shared<MultiHeadAttention>(16, 4, true, true); }, x3, 1e-5f);
        {
            // Dropout on the GPU: Philox masks. Not the CPU's mask, so check
            // what a mask must be: the drop rate, the scale, the gradient,
            // and exact reproduction under a seed
            Dropout drop(0.3f);
            auto ones = G(std::make_shared<Tensor>(Shape{100000}, std::vector<float>(100000, 1.0f)));
            ones->set_requires_grad(true);
            Tensor::manual_seed(4);
            auto y = drop.forward(ones);
            const auto v = y->to_vector();
            int dropped = 0, scaled = 0;
            for (float e : v) { dropped += e == 0.0f; scaled += std::fabs(e - 1.0f / 0.7f) < 1e-6f; }
            CHECK_NEAR("dropout rate on the GPU (100k draws)", dropped / 100000.0f, 0.3f, 0.01f);
            CHECK_TRUE("kept units scaled by 1/(1-rate)", dropped + scaled == 100000);
            sum(y, -1)->backward();
            const auto gv = ones->grad_doubles();
            bool same = true;
            for (size_t i = 0; i < v.size(); i++) same = same && std::fabs(gv[i] - v[i]) < 1e-6;
            CHECK_TRUE("dropout gradient is the mask", same);
            Tensor::manual_seed(4);
            CHECK_TRUE("a seed reproduces the GPU mask", max_diff(drop.forward(ones), y) == 0.0f);
            Tensor::manual_seed(5);
            CHECK_TRUE("another seed gives another mask", max_diff(drop.forward(ones), y) > 0.0f);
        }
        auto ids = std::make_shared<Tensor>(Shape{2, 5}, DType::Int64);
        ids->set_data({1, 4, 4, 0, 9, 3, 3, 7, 2, 1});
        layer_check("Embedding", [] { return std::make_shared<Embedding>(10, 6); }, ids, 1e-6f);
        auto pe_in = random_tensor({5, 6}, 51);
        // ── phase 2: vision, normalisation, recurrent ──
        auto img = random_tensor({2, 3, 9, 9}, 53);
        layer_check("Conv2D k3 s1 p1", [] { return std::make_shared<Conv2D>(3, 5, 3, 1, 1); }, img, 1e-5f);
        layer_check("Conv2D k3 s2 p0", [] { return std::make_shared<Conv2D>(3, 4, 3, 2, 0); }, img, 1e-5f);
        layer_check("ConvTranspose2D k4 s2 p1", [] { return std::make_shared<ConvTranspose2D>(3, 4, 4, 2, 1); }, img, 1e-5f);
        layer_check("MaxPool2D k2", [] { return std::make_shared<MaxPool2D>(2); }, img, 0.0f);
        layer_check("MaxPool2D k3 s2 (overlapping)", [] { return std::make_shared<MaxPool2D>(3, 2); }, img, 0.0f);
        layer_check("AvgPool2D k3", [] { return std::make_shared<AvgPool2D>(3); }, img, 1e-6f);
        layer_check("GroupNorm 3 groups", [] { return std::make_shared<GroupNorm>(3, 6); }, random_tensor({2, 6, 4, 4}, 54), 1e-5f);
        layer_check("Flatten", [] { return std::make_shared<Flatten>(); }, img, 0.0f);
        {
            // BatchNorm: output, gradients, and the running estimate it updates
            std::shared_ptr<BatchNorm> keep_c, keep_g;
            layer_check("BatchNorm (training)", [&] {
                auto b = std::make_shared<BatchNorm>(3);
                (keep_c ? keep_g : keep_c) = b;
                return b;
            }, img, 1e-5f);
            float m = 0;
            for (int c = 0; c < 3; c++)
                m = std::max(m, std::max(std::fabs(keep_g->running_mean[c] - keep_c->running_mean[c]),
                                         std::fabs(keep_g->running_var[c] - keep_c->running_var[c])));
            CHECK_NEAR("BatchNorm running statistics", m, 0.0f, 1e-5f);
            layer_check("BatchNorm (eval)", [] {
                auto b = std::make_shared<BatchNorm>(3);
                b->running_mean = {0.1f, -0.2f, 0.3f};
                b->running_var = {1.5f, 0.7f, 2.0f};
                b->eval();
                return b;
            }, img, 1e-6f);
        }
        auto seq = random_tensor({6, 3, 5}, 55);
        layer_check("RNN", [] { return std::make_shared<RNN>(5, 7); }, seq, 1e-5f);
        layer_check("LSTM", [] { return std::make_shared<LSTM>(5, 7); }, seq, 1e-5f);
        layer_check("GRU", [] { return std::make_shared<GRU>(5, 7); }, seq, 1e-5f);

        // the losses written with raw loops on the CPU
        auto pa = random_tensor({4, 6}, 56), pb = random_tensor({4, 6}, 57), pn = random_tensor({4, 6}, 58);
        auto loss_pair = [&](const char* what, const std::function<TensorPtr(TensorPtr, TensorPtr, TensorPtr)>& f) {
            auto a = pa->detach(), ag = G(pa);
            a->set_requires_grad(true);
            ag->set_requires_grad(true);
            auto lc = f(a, pb, pn), lgg = f(ag, G(pb), G(pn));
            lc->backward();
            lgg->backward();
            double m = 0;
            const auto x1 = ag->grad_doubles(), x2 = a->grad_doubles();
            for (size_t j = 0; j < x2.size(); j++) m = std::max(m, std::fabs(x1[j] - x2[j]));
            CHECK_NEAR((std::string(what) + ": value").c_str(),
                       (float)std::fabs(lgg->item_double() - lc->item_double()), 0.0f, 1e-5f);
            CHECK_NEAR((std::string(what) + ": gradient").c_str(), (float)m, 0.0f, 1e-5f);
        };
        loss_pair("huber_loss", [](TensorPtr a, TensorPtr b, TensorPtr) { return huber_loss(a, b, 0.3f); });
        loss_pair("contrastive_loss", [](TensorPtr a, TensorPtr b, TensorPtr n) { return contrastive_loss(a, b, n, 5.0f); });
        loss_pair("l1_regularization", [](TensorPtr a, TensorPtr, TensorPtr) { return l1_regularization({a}, 0.1f); });
        loss_pair("l2_regularization", [](TensorPtr a, TensorPtr, TensorPtr) { return l2_regularization({a}, 0.1f); });
        loss_pair("mse_loss", [](TensorPtr a, TensorPtr b, TensorPtr) { return mse_loss(a, b); });
        loss_pair("bce_loss", [](TensorPtr a, TensorPtr b, TensorPtr) { return bce_loss(sigmoid(a), divide(b, 2.5f)); });

        PositionalEncoding pe(16, 6);
        CHECK_NEAR("PositionalEncoding", rel(pe.forward(G(pe_in)), pe.forward(pe_in)), 0.0f, 1e-6f);

        // cross-entropy: value and gradient
        auto logits = random_tensor({6, 11}, 52), lg = G(logits);
        logits->set_requires_grad(true);
        lg->set_requires_grad(true);
        const std::vector<int> tg = {0, 3, 10, 5, 5, 7};
        auto lc = cross_entropy_sparse_seq(logits, tg), lgg = cross_entropy_sparse_seq(lg, tg);
        CHECK_NEAR("cross-entropy value", (float)std::fabs(lgg->item_double() - lc->item_double()), 0.0f, 1e-6f);
        lc->backward();
        lgg->backward();
        double m = 0;
        const auto a = lg->grad_doubles(), b = logits->grad_doubles();
        for (size_t j = 0; j < b.size(); j++) m = std::max(m, std::fabs(a[j] - b[j]));
        CHECK_NEAR("cross-entropy gradient", (float)m, 0.0f, 1e-6f);
    }

    // ── the milestone: a GPT's training step on the GPU matches the CPU ──
    {
        Tensor::manual_seed(3);
        GPT cpu(50, 16, 32, 4, 2);
        Tensor::manual_seed(3);
        GPT gpu(50, 16, 32, 4, 2);
        gpu.to(Device::CUDA);
        auto ids = std::make_shared<Tensor>(Shape{2, 12}, DType::Int64);
        std::vector<float> v(24);
        std::vector<int> targets(24);
        for (int i = 0; i < 24; i++) { v[i] = (float)((i * 7 + 3) % 50); targets[i] = (i * 11 + 5) % 50; }
        ids->set_data(v);
        auto lc = cross_entropy_sparse_seq(cpu.forward(ids), targets);
        auto lgpu = cross_entropy_sparse_seq(gpu.forward(ids->to(Device::CUDA)), targets);
        CHECK_NEAR("GPT loss on the GPU", (float)std::fabs(lgpu->item_double() - lc->item_double()), 0.0f, 1e-5f);
        lc->backward();
        lgpu->backward();
        double m = 0, s = 0;
        auto pc = cpu.parameters(), pg = gpu.parameters();
        bool all_on_gpu = true;
        for (size_t i = 0; i < pc.size(); i++) {
            all_on_gpu = all_on_gpu && pg[i]->device() == Device::CUDA;
            const auto a = pg[i]->grad_doubles(), b = pc[i]->grad_doubles();
            for (size_t j = 0; j < b.size(); j++) { m = std::max(m, std::fabs(a[j] - b[j])); s = std::max(s, std::fabs(b[j])); }
        }
        CHECK_TRUE("GPT: every parameter lives on the GPU", all_on_gpu);
        CHECK_NEAR("GPT: every parameter's gradient matches the CPU", (float)(m / std::max(s, 1e-12)), 0.0f, 1e-4f);
    }

    // ── optimizers on the GPU ──
    {
        // run `steps` updates with fixed gradients on both devices, compare weights
        auto opt_check = [&](const char* what, auto make_opt, int steps, float tol) {
            auto w0 = random_tensor({300}, 61);
            auto wc = w0->detach(), wg = G(w0);
            wc->set_requires_grad(true);
            wg->set_requires_grad(true);
            auto oc = make_opt(std::vector<TensorPtr>{wc});
            auto og = make_opt(std::vector<TensorPtr>{wg});
            for (int s = 0; s < steps; s++) {
                std::vector<float> g(300);
                for (int j = 0; j < 300; j++) g[j] = std::sin(0.37f * j + 1.3f * s);
                wc->set_grad(g);
                wg->set_grad(g);
                oc->step();
                og->step();
            }
            CHECK_NEAR(what, rel(wg, wc), 0.0f, tol);
        };
        opt_check("SGD with momentum, 20 steps", [](std::vector<TensorPtr> p) { return std::make_shared<SGD>(p, 0.05f, 0.9f); }, 20, 1e-6f);
        opt_check("SGD, 20 steps", [](std::vector<TensorPtr> p) { return std::make_shared<SGD>(p, 0.05f); }, 20, 1e-6f);
        opt_check("Adam, 20 steps", [](std::vector<TensorPtr> p) { return std::make_shared<Adam>(p, 0.01f); }, 20, 1e-6f);
        opt_check("AdamW, 20 steps", [](std::vector<TensorPtr> p) { return std::make_shared<AdamW>(p, 0.01f); }, 20, 1e-6f);
        opt_check("RMSprop, 20 steps", [](std::vector<TensorPtr> p) { return std::make_shared<RMSprop>(p, 0.01f); }, 20, 1e-6f);

        // gradient clipping: the same norm and the same clipped gradients
        auto pc = random_tensor({1000}, 62), pg = G(pc);
        pc->set_requires_grad(true);
        pg->set_requires_grad(true);
        std::vector<float> g(1000);
        for (int j = 0; j < 1000; j++) g[j] = std::cos(0.11f * j) * 3.0f;
        pc->set_grad(g);
        pg->set_grad(g);
        std::vector<TensorPtr> lc{pc}, lg{pg};
        const float nc = clip_grad_norm(lc, 1.0f), ng = clip_grad_norm(lg, 1.0f);
        CHECK_NEAR("clip_grad_norm: the norm", std::fabs(nc - ng) / nc, 0.0f, 1e-6f);
        double m = 0;
        const auto a = pg->grad_doubles(), b = pc->grad_doubles();
        for (size_t j = 0; j < b.size(); j++) m = std::max(m, std::fabs(a[j] - b[j]));
        CHECK_NEAR("clip_grad_norm: the clipped gradients", (float)m, 0.0f, 1e-6f);
    }

    // ── the milestone: training a GPT on the GPU tracks the CPU step for step ──
    {
        Tensor::manual_seed(8);
        GPT cpu(40, 16, 32, 4, 2);
        Tensor::manual_seed(8);
        GPT gpu(40, 16, 32, 4, 2);
        AdamW oc(cpu.parameters(), 3e-3f);
        AdamW og(gpu.parameters(), 3e-3f);    // built before the move: must follow it
        gpu.to(Device::CUDA);
        auto ids = std::make_shared<Tensor>(Shape{2, 12}, DType::Int64);
        std::vector<float> v(24);
        std::vector<int> targets(24);
        for (int i = 0; i < 24; i++) { v[i] = (float)((i * 5 + 1) % 40); targets[i] = (i * 5 + 6) % 40; }
        ids->set_data(v);
        auto idg = ids->to(Device::CUDA);
        float first = 0, last_c = 0, last_g = 0, worst = 0, first_gap = 0;
        for (int step = 0; step < 15; step++) {
            oc.zero_grad();
            og.zero_grad();
            auto lc = cross_entropy_sparse_seq(cpu.forward(ids), targets);
            auto lg = cross_entropy_sparse_seq(gpu.forward(idg), targets);
            lc->backward();
            lg->backward();
            std::vector<TensorPtr> pc = cpu.parameters(), pg = gpu.parameters();
            clip_grad_norm(pc, 1.0f);
            clip_grad_norm(pg, 1.0f);
            oc.step();
            og.step();
            last_c = (float)lc->item_double();
            last_g = (float)lg->item_double();
            const float gap = std::fabs(last_c - last_g) / last_c;
            if (step == 0) { first = last_c; first_gap = gap; }
            worst = std::max(worst, gap);
        }
        std::printf("  GPT, 15 AdamW steps: loss %.4f -> CPU %.4f, GPU %.4f\n", first, last_c, last_g);
        CHECK_TRUE("GPT training on the GPU lowers the loss", last_g < first * 0.8f);
        // A wrong kernel shows at step 0. After that the two devices' float
        // rounding differs by ~1e-6 a step, and Adam's m / sqrt(v) magnifies
        // differences in near-zero gradients, so the curves drift apart
        // slowly: bounded, not exact.
        CHECK_NEAR("GPT step 0: the same loss on both devices", first_gap, 0.0f, 1e-5f);
        CHECK_NEAR("GPT loss curve on the GPU tracks the CPU (15 steps, drift)", worst, 0.0f, 1e-3f);

        // optimizer state round-trips through a file from the GPU
        og.save_state("_test_cuda_adamw.bin");
        AdamW og2(gpu.parameters(), 3e-3f);
        og2.load_state("_test_cuda_adamw.bin");
        std::remove("_test_cuda_adamw.bin");
        CHECK_TRUE("AdamW state saved from the GPU reloads", og2.t == og.t && og2.m.size() == og.m.size() &&
                   og2.m[0] == og.m[0] && og2.v[3] == og.v[3]);
    }

    // ── phase 4: fused backward kernels, and the op form when it is recorded ──
    {
        // second derivative through LayerNorm (and GELU) on the GPU equals the CPU's:
        // with create_graph the op form runs, not the fused kernel
        auto hvp = [](const TensorPtr& x, const TensorPtr& w, LayerNorm& ln) {
            auto y = sum(multiply(gelu(ln.forward(x)), w), -1);
            auto g = autograd::grad(y, {x}, true)[0];
            return autograd::grad(sum(multiply(g, g), -1), {x})[0];
        };
        Tensor::manual_seed(21);
        LayerNorm lc(8);
        Tensor::manual_seed(21);
        LayerNorm lgpu(8);
        lgpu.to(Device::CUDA);
        auto x0 = random_tensor({3, 8}, 91), w0 = random_tensor({3, 8}, 92);
        auto xc = x0->detach(), xg = G(x0);
        xc->set_requires_grad(true);
        xg->set_requires_grad(true);
        CHECK_NEAR("second derivative through LayerNorm + GELU on the GPU",
                   rel(hvp(xg, G(w0), lgpu), hvp(xc, w0, lc)), 0.0f, 1e-4f);
    }

    // ── phase 3: half precision, GemmEx, autocast, GradScaler ──
    {
        std::printf("  tensor cores: %s, bf16 in hardware: %s\n", cuda::has_tensor_cores() ? "yes" : "no",
                    cuda::has_bf16() ? "yes" : "no");
        auto h = cast_op(G(x), DType::Float16), hc = cast_op(x, DType::Float16);
        CHECK_TRUE("float16 stays float16 through add, exp, sum, softmax",
                   add(h, h)->dtype() == DType::Float16 && exp_op(h)->dtype() == DType::Float16 &&
                   sum(h, -1)->dtype() == DType::Float16 && softmax(h)->dtype() == DType::Float16);
        CHECK_TRUE("float16 ops match the CPU's float16 ops",
                   max_diff(add(h, exp_op(h)), add(hc, exp_op(hc))) <= 1e-2f);
        for (DType d : {DType::Float16, DType::BFloat16}) {
            auto a = random_tensor({48, 64}, 81), b = random_tensor({64, 40}, 82);
            auto ac = cast_op(a, d), bc = cast_op(b, d);
            auto r = matmul(cast_op(G(a), d), cast_op(G(b), d));
            CHECK_TRUE((std::string(dtype_name(d)) + " matmul returns " + dtype_name(d)).c_str(), r->dtype() == d);
            // both round a float32-accumulated product of the same rounded inputs
            CHECK_NEAR((std::string(dtype_name(d)) + " matmul matches the CPU").c_str(),
                       rel(r, matmul(ac, bc)), 0.0f, d == DType::Float16 ? 2e-3f : 1e-2f);
            auto a3 = cast_op(G(random_tensor({3, 8, 16}, 83)), d), b3 = cast_op(G(random_tensor({3, 16, 5}, 84)), d);
            CHECK_TRUE((std::string(dtype_name(d)) + " batched matmul").c_str(),
                       rel(matmul(a3, b3), matmul(a3->to(Device::CPU), b3->to(Device::CPU))) < 1e-2f);
        }
        // gradients through a float16 matmul land in the leaf's float32 buffer
        auto w0 = random_tensor({16, 8}, 85), x0 = random_tensor({4, 16}, 86);
        auto wl = cast_op(w0, DType::Float16)->to(Device::CUDA), wc = cast_op(w0, DType::Float16);
        wl->set_requires_grad(true);
        wc->set_requires_grad(true);
        sum(matmul(cast_op(G(x0), DType::Float16), wl), -1)->backward();
        sum(matmul(cast_op(x0, DType::Float16), wc), -1)->backward();
        double m = 0;
        const auto g1 = wl->grad_doubles(), g2 = wc->grad_doubles();
        for (size_t j = 0; j < g2.size(); j++) m = std::max(m, std::fabs(g1[j] - g2[j]) / std::max(1.0, std::fabs(g2[j])));
        CHECK_NEAR("gradient through a float16 matmul", (float)m, 0.0f, 1e-2f);

        // autocast on the GPU: the same rules as the CPU
        Linear lin(16, 8);
        lin.to(Device::CUDA);
        TensorPtr hl, sm;
        {
            Autocast ac(DType::BFloat16);
            hl = lin.forward(G(x0));
            sm = softmax(hl);
        }
        CHECK_TRUE("autocast: Linear on the GPU returns bfloat16", hl->dtype() == DType::BFloat16);
        CHECK_TRUE("autocast: softmax returns float32", sm->dtype() == DType::Float32);
        sum(sm, -1)->backward();
        const auto lg = lin.weights->grad_doubles();
        bool finite = !lg.empty();
        for (double v : lg) finite = finite && std::isfinite(v);
        CHECK_TRUE("autocast: float32 weights on the GPU get finite gradients", finite);

        // GradScaler with GPU parameters
        auto p1 = G(random_tensor({100}, 87));
        p1->set_requires_grad(true);
        GradScaler scaler(1024.0f);
        scaler.scale_loss(sum(multiply(p1, p1), -1))->backward();
        const bool clean = scaler.unscale({p1});
        const auto gp = p1->grad_doubles();
        const auto pv = p1->to(Device::CPU)->to_vector();
        double err = 0;
        for (size_t j = 0; j < gp.size(); j++) err = std::max(err, std::fabs(gp[j] - 2.0 * pv[j]));
        CHECK_TRUE("GradScaler: unscaled GPU gradients are the true ones", clean && err < 1e-4);
        std::vector<float> bad(100, 1.0f);
        bad[7] = INFINITY;
        p1->set_grad(bad);
        CHECK_TRUE("GradScaler: an inf on the GPU is caught", !scaler.unscale({p1}));
        CHECK_TRUE("GradScaler: and zeroed", p1->grad_doubles()[7] == 0.0);
    }

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
