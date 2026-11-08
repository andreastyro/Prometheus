#include "ml/tensor.hpp"
#include "ml/data/dataloader.hpp"
#include "ml/utils/testing.hpp"
#include <stdio.h>
#include <vector>
#include <algorithm>

using namespace std;

// Sum of a tensor's data, used to assert a shuffle is a permutation: a
// reordering preserves the sum, a duplicated or dropped sample does not.
static double sum_of(TensorPtr t) {
    double s = 0; for (float v : t->values()) s += v; return s;
}

int main() {
    Tensor::manual_seed(7);

    // 6 samples, 2 features: values 1..12, labels alternating 0/1
    auto x = make_shared<Tensor>(Shape{6, 2},
        vector<float>{1,2, 3,4, 5,6, 7,8, 9,10, 11,12});
    auto y = make_shared<Tensor>(Shape{6, 1},
        vector<float>{0, 1, 0, 1, 0, 1});
    const double X_SUM = 78.0;   // 1+2+...+12
    const double Y_SUM = 3.0;

    printf("=== no shuffle, batch_size=2 ===\n");
    {
        DataLoader loader(x, y, 2, false);
        int batches = 0; double seen = 0;
        vector<float> first;
        while (loader.has_next()) {
            auto [bx, by] = loader.next_batch();
            if (batches == 0) first = bx->to_vector();
            CHECK_NEAR("batch x cols", bx->shape()[1], 2, 0.0);
            CHECK_NEAR("batch y cols", by->shape()[1], 1, 0.0);
            seen += sum_of(bx);
            batches++;
        }
        CHECK_NEAR("6 samples / batch 2 -> 3 batches", batches, 3, 0.0);
        CHECK_NEAR("every sample seen exactly once", seen, X_SUM, 1e-4);
        CHECK_NEAR("unshuffled first batch starts at 1", first[0], 1.0, 0.0);
        CHECK_NEAR("unshuffled first batch second row", first[2], 3.0, 0.0);
    }

    printf("\n=== ragged last batch, batch_size=4 ===\n");
    {
        DataLoader loader(x, y, 4, false);
        vector<int> rows;
        while (loader.has_next()) { auto [bx, by] = loader.next_batch(); rows.push_back(bx->shape()[0]); }
        CHECK_NEAR("batch count", (double)rows.size(), 2, 0.0);
        CHECK_NEAR("first batch rows", rows[0], 4, 0.0);
        CHECK_NEAR("last batch rows (remainder)", rows[1], 2, 0.0);
    }

    printf("\n=== shuffle is a permutation, not a corruption ===\n");
    {
        DataLoader loader(x, y, 6, true);
        auto [bx, by] = loader.next_batch();
        CHECK_NEAR("all 6 rows returned", bx->shape()[0], 6, 0.0);
        CHECK_NEAR("shuffled x preserves sum", sum_of(bx), X_SUM, 1e-4);
        CHECK_NEAR("shuffled y preserves sum", sum_of(by), Y_SUM, 1e-4);
        // feature pairs must stay intact: every row is (odd, odd+1)
        bool pairs_intact = true;
        for (int r = 0; r < 6; r++)
            if (bx->data()[r * 2 + 1] != bx->data()[r * 2] + 1.0f) pairs_intact = false;
        CHECK_TRUE("shuffle keeps each row's features together", pairs_intact);
    }

    printf("\n=== reset replays from the start ===\n");
    {
        DataLoader loader(x, y, 3, false);
        auto [b1x, b1y] = loader.next_batch();
        vector<float> before = b1x->to_vector();
        loader.reset();
        auto [b2x, b2y] = loader.next_batch();
        CHECK_TRUE("same first batch after reset", before == b2x->to_vector());
        CHECK_TRUE("has_next() true after reset", loader.has_next());
    }

    printf("\n=== data_split 70/30 ===\n");
    {
        auto x2 = make_shared<Tensor>(Shape{10, 2},
            vector<float>{1,2, 3,4, 5,6, 7,8, 9,10, 11,12, 13,14, 15,16, 17,18, 19,20});
        auto y2 = make_shared<Tensor>(Shape{10, 1},
            vector<float>{0,1,0,1,0,1,0,1,0,1});
        const double ALL = 210.0;   // 1+2+...+20

        DataSplit s = data_split(x2, y2, 0.7f, 0.0f, 0.3f, false);
        CHECK_NEAR("x_train rows", s.x_train->shape()[0], 7, 0.0);
        CHECK_NEAR("x_test rows",  s.x_test->shape()[0],  3, 0.0);
        CHECK_NEAR("y_train rows", s.y_train->shape()[0], 7, 0.0);
        CHECK_NEAR("no sample lost", sum_of(s.x_train) + sum_of(s.x_test), ALL, 1e-4);
        CHECK_NEAR("unshuffled train starts at 1", s.x_train->data()[0], 1.0, 0.0);
        CHECK_NEAR("unshuffled test starts at 15", s.x_test->data()[0], 15.0, 0.0);

        printf("\n=== data_split 60/20/20 ===\n");
        DataSplit t = data_split(x2, y2, 0.6f, 0.2f, 0.2f, false);
        CHECK_NEAR("x_train rows", t.x_train->shape()[0], 6, 0.0);
        CHECK_NEAR("x_val rows",   t.x_val->shape()[0],   2, 0.0);
        CHECK_NEAR("x_test rows",  t.x_test->shape()[0],  2, 0.0);
        CHECK_NEAR("three-way split loses nothing",
                   sum_of(t.x_train) + sum_of(t.x_val) + sum_of(t.x_test), ALL, 1e-4);

        printf("\n=== shuffled 60/20/20 ===\n");
        DataSplit u = data_split(x2, y2, 0.6f, 0.2f, 0.2f, true);
        CHECK_NEAR("shuffled x_train rows", u.x_train->shape()[0], 6, 0.0);
        CHECK_NEAR("shuffled x_val rows",   u.x_val->shape()[0],   2, 0.0);
        CHECK_NEAR("shuffled x_test rows",  u.x_test->shape()[0],  2, 0.0);
        CHECK_NEAR("shuffled split loses nothing",
                   sum_of(u.x_train) + sum_of(u.x_val) + sum_of(u.x_test), ALL, 1e-4);
    }

    return test_report();
}
