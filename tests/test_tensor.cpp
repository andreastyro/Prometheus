#include "ml/tensor.hpp"
#include "ml/ops.hpp"
#include <stdio.h>

using namespace std;

int main() {

    // --- Test 1: Create a tensor and set values ---
    printf("=== Test 1: Create & Set ===\n");
    auto a = make_shared<Tensor>(vector<int>{2, 3});
    a->set(0, 0, 1.0f); a->set(0, 1, 2.0f); a->set(0, 2, 3.0f);
    a->set(1, 0, 4.0f); a->set(1, 1, 5.0f); a->set(1, 2, 6.0f);
    a->print();

    // --- Test 2: zeros and ones ---
    printf("\n=== Test 2: zeros & ones ===\n");
    auto z = Tensor::zeros({2, 3});
    printf("zeros:\n"); z->print();
    auto o = Tensor::ones({2, 3});
    printf("ones:\n"); o->print();

    // --- Test 3: fill ---
    printf("\n=== Test 3: fill ===\n");
    auto f = make_shared<Tensor>(vector<int>{2, 3});
    f->fill(7.0f);
    f->print();

    // --- Test 4: add ---
    printf("\n=== Test 4: add ===\n");
    auto b = make_shared<Tensor>(vector<int>{2, 3});
    b->set(0, 0, 9.0f); b->set(0, 1, 8.0f); b->set(0, 2, 7.0f);
    b->set(1, 0, 6.0f); b->set(1, 1, 5.0f); b->set(1, 2, 4.0f);
    auto s = add(a, b);
    s->print();

    // --- Test 5: matmul ---
    // a is {2,3}, c is {3,2} -> result should be {2,2}
    printf("\n=== Test 5: matmul ===\n");
    auto c = make_shared<Tensor>(vector<int>{3, 2});
    c->set(0, 0, 1.0f); c->set(0, 1, 2.0f);
    c->set(1, 0, 3.0f); c->set(1, 1, 4.0f);
    c->set(2, 0, 5.0f); c->set(2, 1, 6.0f);
    auto m = matmul(a, c);
    m->print();

    return 0;
}
