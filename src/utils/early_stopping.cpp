#include "ml/utils/early_stopping.hpp"
#include <limits>

using namespace std;

EarlyStopping::EarlyStopping(int patience, float min_delta){
    this->patience = patience;
    this->min_delta = min_delta; // min improvement allowed
    counter = 0;
    best_loss = numeric_limits<float>::infinity();
    should_stop = false;
}

bool EarlyStopping::step(float val_loss){
    if (val_loss < best_loss - min_delta){ // new loss must be smaller than best loss - min improvement
        best_loss = val_loss;
        counter = 0;
    } else {
        counter++;
        if (counter >= patience)
            should_stop = true;
    }
    return should_stop;
}

void EarlyStopping::reset(){
    counter = 0;
    best_loss = numeric_limits<float>::infinity();
    should_stop = false;
}
