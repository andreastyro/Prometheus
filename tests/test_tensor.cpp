#include "ml/tensor.hpp"
#include "ml/ops.hpp"
#include "ml/utils/testing.hpp"
#include <stdio.h>

using namespace std;

int main() {
    // a = [[1, 2, 3],
    //      [4, 5, 6]]
    printf("=== create & set ===\n");
    auto a = make_shared<Tensor>(Shape{2, 3});
    a->set(0, 0, 1.0f); a->set(0, 1, 2.0f); a->set(0, 2, 3.0f);
    a->set(1, 0, 4.0f); a->set(1, 1, 5.0f); a->set(1, 2, 6.0f);
    CHECK_NEAR("shape dim 0",   a->shape()[0], 2, 0.0);
    CHECK_NEAR("shape dim 1",   a->shape()[1], 3, 0.0);
    CHECK_NEAR("num_el",        a->num_el(), 6, 0.0);
    CHECK_NEAR("get(0,0)",      a->get(0, 0), 1.0, 0.0);
    CHECK_NEAR("get(1,2)",      a->get(1, 2), 6.0, 0.0);
    CHECK_NEAR("row-major data[4]", a->data()[4], 5.0, 0.0);

    printf("\n=== zeros & ones ===\n");
    auto z = Tensor::zeros({2, 3});
    auto o = Tensor::ones({2, 3});
    double zsum = 0, osum = 0;
    for (float v : z->values()) zsum += v;
    for (float v : o->values()) osum += v;
    CHECK_NEAR("sum(zeros)", zsum, 0.0, 0.0);
    CHECK_NEAR("sum(ones)",  osum, 6.0, 0.0);

    printf("\n=== fill ===\n");
    auto f = make_shared<Tensor>(Shape{2, 3});
    f->fill(7.0f);
    double fsum = 0;
    for (float v : f->values()) fsum += v;
    CHECK_NEAR("sum(fill 7)", fsum, 42.0, 0.0);

    printf("\n=== add ===\n");
    // b = [[9, 8, 7], [6, 5, 4]]; a + b is 10 everywhere
    auto b = make_shared<Tensor>(Shape{2, 3});
    b->set(0, 0, 9.0f); b->set(0, 1, 8.0f); b->set(0, 2, 7.0f);
    b->set(1, 0, 6.0f); b->set(1, 1, 5.0f); b->set(1, 2, 4.0f);
    auto s = add(a, b);
    bool all_ten = true;
    for (float v : s->values()) if (std::fabs(v - 10.0f) > 1e-6f) all_ten = false;
    CHECK_TRUE("a + b == 10 everywhere", all_ten);

    printf("\n=== matmul ===\n");
    // c = [[1,2],[3,4],[5,6]];  a @ c = [[22,28],[49,64]]
    auto c = make_shared<Tensor>(Shape{3, 2});
    c->set(0, 0, 1.0f); c->set(0, 1, 2.0f);
    c->set(1, 0, 3.0f); c->set(1, 1, 4.0f);
    c->set(2, 0, 5.0f); c->set(2, 1, 6.0f);
    auto m = matmul(a, c);
    CHECK_NEAR("matmul shape dim 0", m->shape()[0], 2, 0.0);
    CHECK_NEAR("matmul shape dim 1", m->shape()[1], 2, 0.0);
    CHECK_NEAR("matmul [0,0]", m->get(0, 0), 22.0, 1e-4);
    CHECK_NEAR("matmul [0,1]", m->get(0, 1), 28.0, 1e-4);
    CHECK_NEAR("matmul [1,0]", m->get(1, 0), 49.0, 1e-4);
    CHECK_NEAR("matmul [1,1]", m->get(1, 1), 64.0, 1e-4);

    printf("\n=== reshape & transpose ===\n");
    auto r = a->reshape({3, 2});
    CHECK_NEAR("reshape dim 0", r->shape()[0], 3, 0.0);
    CHECK_NEAR("reshape preserves data[3]", r->data()[3], 4.0, 0.0);
    auto t = a->transpose();
    CHECK_NEAR("transpose dim 0", t->shape()[0], 3, 0.0);
    CHECK_NEAR("transpose dim 1", t->shape()[1], 2, 0.0);
    CHECK_NEAR("transpose [2,1] == a[1,2]", t->get(2, 1), 6.0, 0.0);

    printf("\n=== manual_seed reproducibility ===\n");
    Tensor::manual_seed(99);
    auto r1 = Tensor::randn({50});
    Tensor::manual_seed(99);
    auto r2 = Tensor::randn({50});
    bool same = true;
    for (int i = 0; i < 50; i++) if (r1->data()[i] != r2->data()[i]) same = false;
    CHECK_TRUE("same seed gives same randn", same);

    // The graph must be freed once nothing outside it holds the result. Each
    // backward closure used to capture its own output, a cycle reference
    // counting cannot break, so every graph ever built leaked -- 6.3 MB per
    // step on a small model. A weak pointer to an intermediate tells whether
    // the graph went away.
    {
        std::weak_ptr<Tensor> after_backward, never_backpropagated;
        auto x = Tensor::randn({8, 8});
        x->set_requires_grad(true);
        {
            auto y = relu(matmul(x, x));
            after_backward = y;
            mean(multiply(y, y), -1)->backward();
        }
        {
            auto y = relu(matmul(x, x));
            never_backpropagated = y;
            mean(multiply(y, y), -1);
        }
        CHECK_TRUE("graph is freed after backward", after_backward.expired());
        CHECK_TRUE("graph is freed without backward", never_backpropagated.expired());
    }

    return test_report();
}
