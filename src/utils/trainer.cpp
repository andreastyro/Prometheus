#include "ml/utils/trainer.hpp"
#include <stdio.h>

using namespace std;

void train(
    Module& model,
    DataLoader& loader,
    Optimizer& optimizer,
    function<TensorPtr(TensorPtr, TensorPtr)> loss_fn,
    int epochs,
    bool verbose)
{
    for (int epoch = 0; epoch < epochs; epoch++){
        loader.reset();
        float epoch_loss = 0.0f;

        while (loader.has_next()){
            auto [x, y] = loader.next_batch();

            optimizer.zero_grad();
            auto pred = model.forward(x);
            auto loss = loss_fn(pred, y);
            loss->backward();
            optimizer.step();

            epoch_loss += loss->data[0];
        }

        if (verbose)
            printf("Epoch %d | Loss: %.4f\n", epoch + 1, epoch_loss / loader.size());
    }
}
