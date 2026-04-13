#include "ml/utils/trainer.hpp"
#include "ml/metrics/metrics.hpp"
#include <stdio.h>

using namespace std;

TrainHistory train(
    Module& model,
    DataLoader& loader,
    Optimizer& optimizer,
    function<TensorPtr(TensorPtr, TensorPtr)> loss_fn,
    int epochs,
    bool verbose)
{
    TrainHistory history;

    for (int epoch = 0; epoch < epochs; epoch++){
        loader.reset();
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;

        while (loader.has_next()){
            auto [x, y] = loader.next_batch();

            optimizer.zero_grad();
            auto pred = model.forward(x);
            auto loss = loss_fn(pred, y);
            loss->backward();
            optimizer.step();

            epoch_loss += loss->data[0];
            epoch_acc += accuracy(pred, y);
        }

        epoch_loss /= loader.size();
        epoch_acc /= loader.size();

        history.loss.push_back(epoch_loss);
        history.accuracy.push_back(epoch_acc);

        if (verbose)
            printf("Epoch %d | Loss: %.4f | Accuracy: %.4f\n", epoch + 1, epoch_loss, epoch_acc);
    }

    return history;
}
