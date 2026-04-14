#include "ml/metrics/metrics.hpp"
#include "ml/ops.hpp"
#include <cmath>

using namespace std;

static vector<int> get_classes(TensorPtr t){
    int n = t->shape[0];
    int cols = t->shape[1];
    vector<int> classes(n);

    if (cols == 1){
        for (int i = 0; i < n; i++)
            classes[i] = (int)round(t->data[i]);
    } else {
        auto indices = argmax(t);
        for (int i = 0; i < n; i++)
            classes[i] = (int)indices->data[i];
    }

    return classes;
}

float accuracy(TensorPtr pred, TensorPtr target){
    int n = pred->shape[0];
    int correct = 0;

    auto pred_classes = get_classes(pred);
    auto target_classes = get_classes(target);

    for (int i = 0; i < n; i++)
        if (pred_classes[i] == target_classes[i]) correct++;

    return (float)correct / n;
}

TensorPtr confusion_matrix(TensorPtr pred, TensorPtr target){
    int n = pred->shape[0]; // 
    int cols = pred->shape[1];
    int n_classes = (cols == 1) ? 2 : cols;

    auto cm = Tensor::zeros({n_classes, n_classes});
    auto pred_classes = get_classes(pred);
    auto target_classes = get_classes(target);

    for (int i = 0; i < n; i++) {
        cm->data[target_classes[i] * n_classes + pred_classes[i]] += 1; 
    }

    return cm;
}

float precision(TensorPtr pred, TensorPtr target){
    auto cm = confusion_matrix(pred, target);

    int cols = pred->shape[1];
    int n_classes = (cols == 1) ? 2 : cols;

    float total = 0;

    for (int c = 0; c < n_classes; c++){ // iterate over classes 

        float true_positive = cm->data[c * n_classes + c];
        float false_positive = 0;

        for (int row = 0; row < n_classes; row++)
            false_positive += cm->data[row * n_classes + c];

        false_positive -= true_positive;
        total += (true_positive + false_positive > 0) ? true_positive / (true_positive + false_positive) : 0;
    }

    return total / n_classes;
}

float recall(TensorPtr pred, TensorPtr target){
    auto cm = confusion_matrix(pred, target);

    int cols = pred->shape[1];
    int n_classes = (cols == 1) ? 2 : cols;

    float total = 0;

    for (int c = 0; c < n_classes; c++){
        float true_positive = cm->data[c * n_classes + c];
        float false_negative = 0;

        for (int col = 0; col < n_classes; col++)
            false_negative += cm->data[c * n_classes + col];

        false_negative -= true_positive;
        total += (true_positive + false_negative > 0) ? true_positive / (true_positive + false_negative) : 0;
    }

    return total / n_classes;
}

float f1_score(TensorPtr pred, TensorPtr target){
    float p = precision(pred, target);
    float r = recall(pred, target);

    float denom = p + r;
    float f1 = (denom != 0) ? (2 * p * r / denom) : 0;

    return f1;
}

float r2_score(TensorPtr pred, TensorPtr target){

    float ss_res = 0, ss_tot = 0, mean_target = 0;

    for (int i = 0; i < target->num_el() ; i++){
        mean_target += target->data[i];
    }

    mean_target /= target->num_el(); // mean = sum/n;
    
    for (int i = 0; i < pred->shape[0]; i++){
        ss_res += pow(target->data[i] - pred->data[i], 2);
        ss_tot += pow(target->data[i] - mean_target, 2);
    }

    float r2 = 1 - ss_res / ss_tot;

    return r2;
}

TensorPtr predict(Module& model, TensorPtr x){
    auto output = model.forward(x);
    return argmax(output);
}