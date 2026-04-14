#include "ml/tensor.hpp"
#include "ml/metrics/metrics.hpp"
#include "ml/nn/linear.hpp"
#include "ml/nn/activations.hpp"
#include "ml/nn/sequential.hpp"
#include <stdio.h>

using namespace std;

int main(){

    // === Binary classification ===
    // pred: probabilities, target: 0 or 1
    auto bin_pred = make_shared<Tensor>(
        vector<int>{5, 1},
        vector<float>{0.9f, 0.2f, 0.8f, 0.3f, 0.7f}
    );
    auto bin_target = make_shared<Tensor>(
        vector<int>{5, 1},
        vector<float>{1, 0, 1, 0, 0}  // last one is wrong
    );

    printf("=== Binary Classification ===\n");
    printf("accuracy:  %.4f (expected ~0.8)\n", accuracy(bin_pred, bin_target));
    printf("precision: %.4f\n", precision(bin_pred, bin_target));
    printf("recall:    %.4f\n", recall(bin_pred, bin_target));
    printf("f1:        %.4f\n", f1_score(bin_pred, bin_target));

    auto bin_cm = confusion_matrix(bin_pred, bin_target);
    printf("confusion matrix (2x2):\n");
    printf("  TN=%.0f FP=%.0f\n", bin_cm->data[0], bin_cm->data[1]);
    printf("  FN=%.0f TP=%.0f\n", bin_cm->data[2], bin_cm->data[3]);

    // === Multi-class classification ===
    // 4 samples, 3 classes — softmax-style probabilities
    auto mc_pred = make_shared<Tensor>(
        vector<int>{4, 3},
        vector<float>{
            0.8f, 0.1f, 0.1f,  // predicts class 0 → correct
            0.1f, 0.7f, 0.2f,  // predicts class 1 → correct
            0.2f, 0.3f, 0.5f,  // predicts class 2 → correct
            0.6f, 0.2f, 0.2f   // predicts class 0 → wrong (true is 1)
        }
    );
    auto mc_target = make_shared<Tensor>(
        vector<int>{4, 3},
        vector<float>{
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
            0, 1, 0
        }
    );

    printf("\n=== Multi-class Classification ===\n");
    printf("accuracy:  %.4f (expected 0.75)\n", accuracy(mc_pred, mc_target));
    printf("precision: %.4f\n", precision(mc_pred, mc_target));
    printf("recall:    %.4f\n", recall(mc_pred, mc_target));
    printf("f1:        %.4f\n", f1_score(mc_pred, mc_target));

    auto mc_cm = confusion_matrix(mc_pred, mc_target);
    printf("confusion matrix (3x3):\n");
    for (int i = 0; i < 3; i++){
        printf("  ");
        for (int j = 0; j < 3; j++)
            printf("%.0f ", mc_cm->data[i * 3 + j]);
        printf("\n");
    }

    // === Regression ===
    auto reg_pred = make_shared<Tensor>(
        vector<int>{5, 1},
        vector<float>{2.1f, 3.9f, 5.8f, 8.2f, 10.1f}
    );
    auto reg_target = make_shared<Tensor>(
        vector<int>{5, 1},
        vector<float>{2.0f, 4.0f, 6.0f, 8.0f, 10.0f}
    );

    printf("\n=== Regression ===\n");
    printf("r2_score: %.4f (expected close to 1.0)\n", r2_score(reg_pred, reg_target));

    // === predict() ===
    Sequential model({
        new Linear(2, 4),
        new ReLU(),
        new Linear(4, 3),
        new Softmax()
    });

    auto x = make_shared<Tensor>(
        vector<int>{3, 2},
        vector<float>{1, 2, 3, 4, 5, 6}
    );

    printf("\n=== predict() ===\n");
    auto labels = predict(model, x);
    printf("predicted classes: ");
    for (int i = 0; i < 3; i++)
        printf("%.0f ", labels->data[i]);
    printf("\n");

    return 0;
}
