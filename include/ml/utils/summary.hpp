#pragma once
#include "ml/nn/sequential.hpp"
#include "ml/nn/activations.hpp"
#include "ml/nn/linear.hpp"
#include "ml/nn/dropout.hpp"
#include <string>

void model_summary(std::vector<Module*> layers);
