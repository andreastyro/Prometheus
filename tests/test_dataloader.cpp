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

    // 10 samples, 2 features
    auto x2 = make_shared<Tensor>(
        vector<int>{10, 2},
        vector<float>{1,2, 3,4, 5,6, 7,8, 9,10, 11,12, 13,14, 15,16, 17,18, 19,20}
    );
    auto y2 = make_shared<Tensor>(
        vector<int>{10, 1},
        vector<float>{0,1,0,1,0,1,0,1,0,1}
    );

    printf("\n=== Train/Test split (70/30) ===\n");
    DataSplit split = data_split(x2, y2, 0.7f, 0.0f, 0.3f, false);
    printf("x_train rows=%d | x_test rows=%d\n", split.x_train->shape[0], split.x_test->shape[0]);
    printf("x_train: ");
    for (float v : split.x_train->data) printf("%.0f ", v);
    printf("\nx_test:  ");
    for (float v : split.x_test->data) printf("%.0f ", v);
    printf("\n");

    printf("\n=== Train/Val/Test split (60/20/20) ===\n");
    DataSplit split2 = data_split(x2, y2, 0.6f, 0.2f, 0.2f, false);
    printf("x_train rows=%d | x_val rows=%d | x_test rows=%d\n",
        split2.x_train->shape[0], split2.x_val->shape[0], split2.x_test->shape[0]);

    printf("\n=== Shuffled Train/Val/Test split (60/20/20) ===\n");
    DataSplit split3 = data_split(x2, y2, 0.6f, 0.2f, 0.2f, true);
    printf("x_train rows=%d: ", split3.x_train->shape[0]);
    for (float v : split3.x_train->data) printf("%.0f ", v);
    printf("\nx_val   rows=%d: ", split3.x_val->shape[0]);
    for (float v : split3.x_val->data) printf("%.0f ", v);
    printf("\nx_test  rows=%d: ", split3.x_test->shape[0]);
    for (float v : split3.x_test->data) printf("%.0f ", v);
    printf("\n");

    return 0;
}
