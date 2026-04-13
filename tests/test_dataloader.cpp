#include "ml/tensor.hpp"
#include "ml/data/dataloader.hpp"
#include <stdio.h>

using namespace std;

int main() {

    // 6 samples, 2 features each
    auto x = make_shared<Tensor>(
        vector<int>{6, 2},
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    );

    // 6 samples, 1 label each
    auto y = make_shared<Tensor>(
        vector<int>{6, 1},
        vector<float>{0, 1, 0, 1, 0, 1}
    );

    printf("=== No shuffle, batch_size=2 ===\n");
    DataLoader loader(x, y, 2, false);
    int batch = 0;
    while (loader.has_next()) {
        auto [bx, by] = loader.next_batch();
        printf("Batch %d | x rows=%d cols=%d | y rows=%d cols=%d\n",
            ++batch, bx->shape[0], bx->shape[1], by->shape[0], by->shape[1]);
        printf("  x: ");
        for (float v : bx->data) printf("%.0f ", v);
        printf("\n  y: ");
        for (float v : by->data) printf("%.0f ", v);
        printf("\n");
    }

    printf("\n=== Last batch smaller than batch_size (batch_size=4) ===\n");
    DataLoader loader2(x, y, 4, false);
    batch = 0;
    while (loader2.has_next()) {
        auto [bx, by] = loader2.next_batch();
        printf("Batch %d | x rows=%d | y rows=%d\n", ++batch, bx->shape[0], by->shape[0]);
    }

    printf("\n=== Shuffle enabled ===\n");
    DataLoader loader3(x, y, 6, true);
    auto [bx, by] = loader3.next_batch();
    printf("Shuffled x: ");
    for (float v : bx->data) printf("%.0f ", v);
    printf("\nShuffled y: ");
    for (float v : by->data) printf("%.0f ", v);
    printf("\n");

    printf("\n=== Reset test ===\n");
    DataLoader loader4(x, y, 3, false);
    auto [b1x, b1y] = loader4.next_batch();
    printf("Before reset, first batch x: ");
    for (float v : b1x->data) printf("%.0f ", v);
    loader4.reset();
    auto [b2x, b2y] = loader4.next_batch();
    printf("\nAfter reset, first batch x:  ");
    for (float v : b2x->data) printf("%.0f ", v);
    printf("\n");

    return 0;
}
