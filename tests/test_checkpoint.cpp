#include "ml/tensor.hpp"
#include "ml/utils/model_io.hpp"
#include "ml/optim/adam.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    // Create two parameters with known values
    auto p1 = std::make_shared<Tensor>(std::vector<int>{2, 3},
                                       std::vector<float>{1,2,3,4,5,6});
    auto p2 = std::make_shared<Tensor>(std::vector<int>{1, 3},
                                       std::vector<float>{7,8,9});
    std::vector<TensorPtr> params = {p1, p2};

    save_checkpoint("_test_ckpt.bin", params, 42, 0.123f);

    // Overwrite with zeroes so we can verify the load restores them
    p1->fill(0.0f);
    p2->fill(0.0f);

    std::vector<TensorPtr> loaded;
    auto ckpt = load_checkpoint("_test_ckpt.bin", loaded);

    assert(ckpt.epoch == 42    && "epoch must survive round-trip");
    assert(std::fabs(ckpt.loss - 0.123f) < 1e-5f && "loss must survive round-trip");
    assert(loaded.size() == 2  && "must restore both parameters");
    std::vector<int> expected_shape = {2, 3};
    assert(loaded[0]->shape == expected_shape);
    assert(std::fabs(loaded[0]->data[5] - 6.0f) < 1e-5f && "data must survive round-trip");
    assert(std::fabs(loaded[1]->data[2] - 9.0f) < 1e-5f);

    // Adam save/load state round-trip
    auto pa = std::make_shared<Tensor>(std::vector<int>{1, 2},
                                       std::vector<float>{1.0f, 2.0f});
    pa->requires_grad = true;
    pa->grad = {0.1f, 0.2f};

    Adam opt({pa}, 0.01f);
    opt.step();
    opt.save_state("_test_adam_state.bin");

    Adam opt2({pa}, 0.01f);
    opt2.load_state("_test_adam_state.bin");
    assert(opt2.t == opt.t);
    assert(std::fabs(opt2.m[0][0] - opt.m[0][0]) < 1e-6f);
    assert(std::fabs(opt2.v[0][1] - opt.v[0][1]) < 1e-6f);

    printf("test_checkpoint PASSED\n");
    return 0;
}
