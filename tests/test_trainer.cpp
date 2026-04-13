#include "ml/tensor.hpp"
#include "ml/nn/linear.hpp"
#include "ml/nn/activations.hpp"
#include "ml/nn/sequential.hpp"
#include "ml/optim/adam.hpp"
#include "ml/loss.hpp"
#include "ml/data/dataloader.hpp"
#include "ml/utils/trainer.hpp"
#include <stdio.h>

using namespace std;

int main(){

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
    train(model, loader, optimizer, mse_loss, 20, true);

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

    return 0;
}
