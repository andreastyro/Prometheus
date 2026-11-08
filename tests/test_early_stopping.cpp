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
#include "ml/utils/testing.hpp"
#include <cmath>

using namespace std;

int main(){
    Tensor::manual_seed(11);

    // 10 train samples
    auto x_train = make_shared<Tensor>(Shape{10, 2},
        vector<float>{1,2, 3,4, 5,6, 7,8, 9,10, 2,1, 4,3, 6,5, 8,7, 10,9}
    );
    auto y_train = make_shared<Tensor>(Shape{10, 1},
        vector<float>{0,1,0,1,0,1,0,1,0,1}
    );

    // 4 val samples
    auto x_val = make_shared<Tensor>(Shape{4, 2},
        vector<float>{1,3, 2,4, 5,7, 6,8}
    );
    auto y_val = make_shared<Tensor>(Shape{4, 1},
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

    int    epochs_run = 0;
    bool   stopped    = false;
    float  last_val   = 0.0f;

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

        printf("Epoch %d | Val Loss: %.4f\n", epoch + 1, val_loss->data()[0]);

        epochs_run = epoch + 1;
        last_val   = val_loss->data()[0];

        if (es.step(val_loss->data()[0])){
            printf("Early stopping triggered at epoch %d\n", epoch + 1);
            stopped = true;
            break;
        }
    }

    CHECK_TRUE("early stopping fired before the 100-epoch cap", stopped);
    CHECK_TRUE("ran more epochs than the patience of 5", epochs_run > 5);
    CHECK_TRUE("stopped before the cap", epochs_run < 100);
    CHECK_TRUE("validation loss is finite", std::isfinite(last_val));

    // patience semantics: a monotonically worsening loss must stop in
    // exactly patience+1 steps, and an improving one must never stop.
    {
        EarlyStopping worse(3, 0.0f);
        int n = 0; bool fired = false;
        for (float v = 1.0f; n < 20; v += 1.0f) { n++; if (worse.step(v)) { fired = true; break; } }
        CHECK_TRUE("worsening loss triggers stop", fired);
        CHECK_NEAR("stops after patience+1 worsening steps", n, 4, 0.0);

        EarlyStopping better(3, 0.0f);
        bool fired2 = false;
        for (float v = 10.0f; v > 0.0f; v -= 1.0f) if (better.step(v)) fired2 = true;
        CHECK_TRUE("improving loss never triggers stop", !fired2);
    }

    return test_report();
}
