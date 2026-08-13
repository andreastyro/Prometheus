#include "ml/nn/vision/residual_block.hpp"
#include "ml/ops.hpp"

using namespace std;

ResidualBlock::ResidualBlock(int in_channels, int out_channels, int stride, int num_groups)
    : conv1(in_channels,  out_channels, 3, stride, 1, "kaiming"),
      norm1(num_groups, out_channels),
      conv2(out_channels, out_channels, 3, 1,      1, "kaiming"),
      norm2(num_groups, out_channels) {

    if (in_channels != out_channels || stride != 1)
        proj = make_unique<Conv2D>(in_channels, out_channels, 1, stride, 0, "kaiming");
}

TensorPtr ResidualBlock::forward(TensorPtr input) {
    auto residual = input;

    auto x = relu(norm1.forward(conv1.forward(input)));
    x = norm2.forward(conv2.forward(x));

    if (proj)
        residual = proj->forward(residual);

    return relu(add(x, residual));
}

vector<TensorPtr> ResidualBlock::parameters() {
    auto params = conv1.parameters();
    auto n1 = norm1.parameters();
    auto c2 = conv2.parameters();
    auto n2 = norm2.parameters();
    params.insert(params.end(), n1.begin(), n1.end());
    params.insert(params.end(), c2.begin(), c2.end());
    params.insert(params.end(), n2.begin(), n2.end());
    if (proj) {
        auto p = proj->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}
