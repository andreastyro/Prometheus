#include "ml/tensor.hpp"
#include "ml/optim/adamw.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    // Single parameter, gradient = 1.0, weight = 0.5
    auto p = std::make_shared<Tensor>(std::vector<int>{1, 1}, std::vector<float>{0.5f});
    p->requires_grad = true;
    p->grad = {1.0f};

    float lr = 0.1f, wd = 0.01f;
    AdamW opt({p}, lr, 0.9f, 0.999f, 1e-8f, wd);

    float w_before = p->data[0];
    opt.step();

    // Weight must decrease (gradient pushes it down, weight decay also shrinks it)
    assert(p->data[0] < w_before && "AdamW should decrease weight when gradient > 0");

    // Weight decay test: zero gradient → only decay term acts
    p->data[0] = 1.0f;
    p->grad = {0.0f};
    // Rebuild fresh optimizer so t starts at 0 again
    AdamW opt2({p}, lr, 0.9f, 0.999f, 1e-8f, wd);
    float w2 = p->data[0];
    opt2.step();
    // With zero gradient and weight_decay > 0, weight should shrink toward 0
    assert(p->data[0] < w2 && "weight decay alone should shrink weight");

    // Round-trip save/load state
    AdamW saver({p}, lr, 0.9f, 0.999f, 1e-8f, wd);
    p->grad = {0.5f};
    saver.step();
    saver.save_state("_test_adamw_state.bin");

    AdamW loader({p}, lr, 0.9f, 0.999f, 1e-8f, wd);
    loader.load_state("_test_adamw_state.bin");
    assert(loader.t == saver.t && "t must survive save/load");
    assert(std::fabs(loader.m[0][0] - saver.m[0][0]) < 1e-6f && "m must survive save/load");
    assert(std::fabs(loader.v[0][0] - saver.v[0][0]) < 1e-6f && "v must survive save/load");

    printf("test_adamw PASSED\n");
    return 0;
}
