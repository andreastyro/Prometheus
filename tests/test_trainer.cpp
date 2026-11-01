#include "ml/tensor.hpp"
#include "ml/nn/linear.hpp"
#include "ml/nn/activations.hpp"
#include "ml/nn/sequential.hpp"
#include "ml/optim/adam.hpp"
#include "ml/loss.hpp"
#include "ml/data/dataloader.hpp"
#include "ml/utils/trainer.hpp"
#include "ml/utils/testing.hpp"
#include <stdio.h>
#include <cmath>

using namespace std;

int main(){
    Tensor::manual_seed(42);

    // 6 samples, 2 features
    auto x = make_shared<Tensor>(
        vector<int>{6, 2},
        vector<float>{1,2, 3,4, 5,6, 7,8, 9,10, 11,12}
    );
    auto y = make_shared<Tensor>(
        vector<int>{6, 1},
        vector<float>{0, 1, 0, 1, 0, 1}
    );

    // regression model (no sigmoid)
    Sequential model({
        new Linear(2, 4),
        new ReLU(),
        new Linear(4, 1)
    });

    DataLoader loader(x, y, 2, false);
    Adam optimizer(model.parameters(), 0.01f);

    printf("=== Regression (MSE, no sigmoid) ===\n");
    auto reg = train(model, loader, optimizer, mse_loss, 20, true);
    CHECK_NEAR("regression: 20 epochs recorded", (double)reg.loss.size(), 20, 0.0);
    CHECK_TRUE("regression: loss is finite", std::isfinite(reg.loss.back()));
    CHECK_TRUE("regression: loss decreased", reg.loss.back() < reg.loss.front());

    // classification model (with sigmoid output)
    printf("\n=== Classification (BCE, sigmoid output) ===\n");
    Sequential clf({
        new Linear(2, 4),
        new ReLU(),
        new Linear(4, 1),
        new Sigmoid()
    });

    DataLoader loader2(x, y, 2, false);
    Adam optimizer2(clf.parameters(), 0.01f);

    auto history = train(clf, loader2, optimizer2, bce_loss, 50, true);

    printf("\nFinal loss:     %.4f\n", history.loss.back());
    printf("Final accuracy: %.4f\n", history.accuracy.back());

    CHECK_NEAR("classification: 50 epochs recorded", (double)history.loss.size(), 50, 0.0);
    CHECK_TRUE("classification: loss is finite", std::isfinite(history.loss.back()));
    CHECK_TRUE("classification: loss decreased", history.loss.back() < history.loss.front());
    CHECK_TRUE("accuracy in [0, 1]",
               history.accuracy.back() >= 0.0f && history.accuracy.back() <= 1.0f);

    // bce_loss must survive a saturated sigmoid rather than throwing on log(0)
    auto sat = make_shared<Tensor>(vector<int>{1, 1}, vector<float>{1.0f});
    auto tgt = make_shared<Tensor>(vector<int>{1, 1}, vector<float>{0.0f});
    bool ok = true;
    try { auto l = bce_loss(sat, tgt); ok = std::isfinite(l->data[0]); }
    catch (const std::exception&) { ok = false; }
    CHECK_TRUE("bce_loss(p=1, y=0) is finite, not a throw", ok);

    return test_report();
}
