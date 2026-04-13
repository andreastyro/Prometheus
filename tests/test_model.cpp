#include "ml/tensor.hpp"
#include "ml/ops.hpp"
#include "ml/loss.hpp"
#include "ml/nn/linear.hpp"
#include "ml/nn/activations.hpp"
#include "ml/nn/sequential.hpp"
#include "ml/optim/adam.hpp"
#include <stdio.h>

using namespace std;

int main() {

    // --- Build model: 2 inputs -> 4 hidden -> 1 output ---
    Sequential model({
        new Linear(2, 4),
        new ReLU(),
        new Linear(4, 1)
    });

    // --- Dummy input: 3 samples, 2 features each ---
    auto input = make_shared<Tensor>(
        vector<int>{3, 2},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
    );
    input->requires_grad = false;

    // --- Target values ---
    auto target = make_shared<Tensor>(
        vector<int>{3, 1},
        vector<float>{1.0f, 0.0f, 1.0f}
    );

    // --- Optimizer ---
    Adam optimizer(model.parameters(), 0.01f);

    // --- Training loop: 10 steps ---
    printf("Training...\n");
    for (int step = 0; step < 100; step++) {
        optimizer.zero_grad();

        auto pred = model.forward(input);
        auto loss = mse_loss(pred, target);

        loss->backward();
        optimizer.step();

        printf("Step %d | Loss: %.4f\n", step + 1, loss->data[0]);
    }

    // --- Final prediction ---
    printf("\nFinal predictions:\n");
    auto pred = model.forward(input);
    pred->print();

    // --- Manual model (no Sequential) ---
    printf("\n=== Manual model (no Sequential) ===\n");
    Linear l1(2, 4);
    ReLU relu;
    Linear l2(4, 1);

    auto params = l1.parameters();
    auto p2 = l2.parameters();
    params.insert(params.end(), p2.begin(), p2.end());

    Adam optimizer2(params, 0.01f);

    for (int step = 0; step < 100; step++){
        optimizer2.zero_grad();

        auto h = relu.forward(l1.forward(input));
        auto out = l2.forward(h);
        auto loss = mse_loss(out, target);

        loss->backward();
        optimizer2.step();

        if ((step + 1) % 20 == 0)
            printf("Step %d | Loss: %.4f\n", step + 1, loss->data[0]);
    }

    printf("\nFinal predictions (manual):\n");
    auto out = l2.forward(relu.forward(l1.forward(input)));
    out->print();

    return 0;
}
