#include "ml/metrics/metrics.hpp"
#include <cmath>

using namespace std;

float accuracy(TensorPtr pred, TensorPtr target){
    int n = pred->shape[0];
    int cols = pred->shape[1];
    int correct = 0;

    for (int i = 0; i < n; i++){
        int pred_class, target_class;

        if (cols == 1){
            // binary classification — round to 0 or 1
            pred_class = round(pred->data[i]);
            target_class = round(target->data[i]);
        } else {
            // multi-class — argmax across columns
            pred_class = 0;
            target_class = 0;
            float max_pred = pred->data[i * cols];
            float max_target = target->data[i * cols];

            for (int j = 1; j < cols; j++){
                if (pred->data[i * cols + j] > max_pred){
                    max_pred = pred->data[i * cols + j];
                    pred_class = j;
                }
                if (target->data[i * cols + j] > max_target){
                    max_target = target->data[i * cols + j];
                    target_class = j;
                }
            }
        }

        if (pred_class == target_class) correct++;
    }

    return (float)correct / n;
}
