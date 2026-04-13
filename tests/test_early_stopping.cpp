#include "ml/tensor.hpp"
#include "ml/nn/linear.hpp"
#include "ml/nn/activations.hpp"
#include "ml/nn/sequential.hpp"
#include "ml/optim/adam.hpp"
#include "ml/loss.hpp"
#include "ml/data/dataloader.hpp"
#include "ml/utils/early_stopping.hpp"
#include "ml/metrics/metrics.hpp"
#include <stdio.h>

using namespace std;

int main(){

    // 10 train samples
    auto x_train = make_shared<Tensor>(
        vector<int>{10, 2},
        vector<float>{1,2, 3,4, 5,6, 7,8, 9,10, 2,1, 4,3, 6,5, 8,7, 10,9}
    );
    auto y_train = make_shared<Tensor>(
        vector<int>{10, 1},
        vector<float>{0,1,0,1,0,1,0,1,0,1}
    );

    // 4 val samples
    auto x_val = make_shared<Tensor>(
        vector<int>{4, 2},
        vector<float>{1,3, 2,4, 5,7, 6,8}
    );
    auto y_val = make_shared<Tensor>(
        vector<int>{4, 1},
        vector<float>{0,1,0,1}
    );

    Sequential model({
        new Linear(2, 4, "kaiming"),
        new ReLU(),
        new Linear(4, 1),
        new Sigmoid()
    });

    DataLoader train_loader(x_train, y_train, 5, false);
    DataLoader val_loader(x_val, y_val, 4, false);
    Adam optimizer(model.parameters(), 0.01f);
    EarlyStopping es(5, 0.001f);

    printf("Training with early stopping (patience=5)...\n");

    for (int epoch = 0; epoch < 100; epoch++){
        train_loader.reset();

        // train
        while (train_loader.has_next()){
            auto [bx, by] = train_loader.next_batch();
            optimizer.zero_grad();
            auto pred = model.forward(bx);
            auto loss = bce_loss(pred, by);
            loss->backward();
            optimizer.step();
        }

        // validate
        val_loader.reset();
        auto [vx, vy] = val_loader.next_batch();
        auto val_pred = model.forward(vx);
        auto val_loss = bce_loss(val_pred, vy);

        printf("Epoch %d | Val Loss: %.4f\n", epoch + 1, val_loss->data[0]);

        if (es.step(val_loss->data[0])){
            printf("Early stopping triggered at epoch %d\n", epoch + 1);
            break;
        }
    }

    return 0;
}
