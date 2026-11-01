// Numeric gradient check over every layer, op and loss in the library,
// driven by the public gradcheck() API.

#include "ml/tensor.hpp"
#include "ml/ops.hpp"
#include "ml/loss.hpp"
#include "ml/nn/linear.hpp"
#include "ml/nn/groupnorm.hpp"
#include "ml/nn/batchnorm.hpp"
#include "ml/nn/nlp/layernorm.hpp"
#include "ml/nn/nlp/embedding.hpp"
#include "ml/nn/nlp/attention.hpp"
#include "ml/nn/nlp/positional_encoding.hpp"
#include "ml/nn/nlp/transformer.hpp"
#include "ml/nn/rnn/rnn.hpp"
#include "ml/nn/rnn/lstm.hpp"
#include "ml/nn/rnn/gru.hpp"
#include "ml/nn/vision/conv2d.hpp"
#include "ml/nn/vision/convtranspose2d.hpp"
#include "ml/nn/vision/maxpool2d.hpp"
#include "ml/nn/vision/avgpool2d.hpp"
#include "ml/nn/vision/residual_block.hpp"
#include "ml/utils/gradcheck.hpp"
#include "ml/utils/testing.hpp"

#include <vector>
#include <string>

using namespace std;

static GradCheckOptions OPTS;

// Run one check and fold the result into the pass/fail accounting of the suite.
static void check(const string& name,
                  function<TensorPtr()> f,
                  vector<TensorPtr> inputs,
                  GradCheckOptions o = OPTS) {
    GradCheckResult r = gradcheck(name, f, inputs, o);
    printf("  %-44s %3d checked, %d failed\n", name.c_str(), r.checked, r.failed);
    if (!r.passed) printf("%s", r.report.c_str());
    CHECK_TRUE((name + " gradients").c_str(), r.passed);
}

// Known-broken checks. These layers build a correct forward pass but assemble
// their result into a raw buffer, which detaches it from the graph they built,
// so every analytical gradient comes back exactly 0. Marking them xfail keeps
// the suite usable as a regression gate while the bugs are tracked. The mark is
// strict: if one starts passing, that is reported as a failure so the entry gets
// removed rather than silently masking a fix.
static void check_xfail(const string& name,
                        const string& why,
                        function<TensorPtr()> f,
                        vector<TensorPtr> inputs,
                        GradCheckOptions o = OPTS) {
    GradCheckResult r = gradcheck(name, f, inputs, o);
    if (r.passed) {
        printf("  %-44s XPASS - now correct, remove the xfail\n", name.c_str());
        CHECK_TRUE((name + " is marked broken but now passes").c_str(), false);
    } else {
        printf("  %-44s XFAIL %d/%d (%s)\n",
               name.c_str(), r.failed, r.checked, why.c_str());
    }
}

static TensorPtr tin(vector<int> shape, bool grad = true) {
    auto t = make_shared<Tensor>(shape);
    // offset by 0.05 so no element lands on 0, where relu/mae/huber have a
    // kink and a central difference straddles the non-differentiable point
    for (int i = 0; i < t->num_el(); i++) t->data[i] = 0.1f * ((i % 7) - 3) + 0.05f;
    t->requires_grad = grad;
    if (grad) t->grad.assign(t->num_el(), 0.0f);
    return t;
}

// Probabilities strictly inside (0, 1), for the losses that take them.
static TensorPtr tprob(vector<int> shape) {
    auto t = make_shared<Tensor>(shape);
    for (int i = 0; i < t->num_el(); i++) t->data[i] = 0.2f + 0.1f * (i % 6);
    t->requires_grad = true;
    t->grad.assign(t->num_el(), 0.0f);
    return t;
}

int main() {
    Tensor::manual_seed(1234);

    printf("\n=== ops ===\n");
    {
        auto x = tin({3, 4});
        check("gelu",         [&]{ return gelu(x); },               {x});
        check("relu",         [&]{ return relu(x); },               {x});
        check("sigmoid",      [&]{ return sigmoid(x); },            {x});
        check("tanh_op",      [&]{ return tanh_op(x); },            {x});
        check("softmax",      [&]{ return softmax(x); },            {x});
        check("transpose_op", [&]{ return transpose_op(x); },       {x});
        check("reshape_op",   [&]{ return reshape_op(x, {4, 3}); }, {x});
        check("exp_op",       [&]{ return exp_op(x); },             {x});

        auto y = tin({3, 4});
        check("add",      [&]{ return add(x, y); },      {x, y});
        check("multiply", [&]{ return multiply(x, y); }, {x, y});
        check("subtract", [&]{ return subtract(x, y); }, {x, y});

        auto m = tin({4, 2});
        check("matmul", [&]{ return matmul(x, m); }, {x, m});
    }

    printf("\n=== losses ===\n");
    {
        auto pred = tin({3, 2});
        auto targ = tin({3, 2}, false);
        check("mse_loss",            [&]{ return mse_loss(pred, targ); },            {pred});
        check("mae_loss",            [&]{ return mae_loss(pred, targ); },            {pred});
        check("huber_loss",          [&]{ return huber_loss(pred, targ); },          {pred});
        check("reconstruction_loss", [&]{ return reconstruction_loss(pred, targ); }, {pred});

        auto p = tprob({3, 2});
        auto q = tprob({3, 2});
        for (int i = 0; i < q->num_el(); i++) q->data[i] = 1.0f - q->data[i];  // differ from p
        q->requires_grad = false;
        check("bce_loss",      [&]{ return bce_loss(p, q); },      {p});
        check_xfail("kl_divergence", "computed on raw floats, no grad node", [&]{ return kl_divergence(p, q); }, {p});

        auto logits = tprob({2, 4});
        auto onehot = make_shared<Tensor>(vector<int>{2, 4},
                                          vector<float>{1, 0, 0, 0, 0, 0, 1, 0});
        check("cross_entropy_loss",
              [&]{ return cross_entropy_loss(logits, onehot); }, {logits});

        auto lg = tin({3, 5});
        check("cross_entropy_sparse_seq",
              [&]{ return cross_entropy_sparse_seq(lg, {0, 2, 4}); }, {lg});

        auto a = tin({1, 3}), pos = tin({1, 3}), neg = tin({1, 3});
        check("contrastive_loss",
              [&]{ return contrastive_loss(a, pos, neg); }, {a, pos, neg});
    }

    printf("\n=== dense / norm layers ===\n");
    {
        Linear fc(4, 3);
        auto x = tin({2, 4});
        check("Linear", [&]{ return fc.forward(x); }, {fc.weights, fc.bias, x});

        LayerNorm ln(4);
        auto xn = tin({3, 4});
        check("LayerNorm", [&]{ return ln.forward(xn); },
              {ln.parameters()[0], ln.parameters()[1], xn});

        GroupNorm gn(2, 4);
        auto xg = tin({1, 4, 2, 2});
        check_xfail("GroupNorm", "no backward implemented at all", [&]{ return gn.forward(xg); },
              {gn.parameters()[0], gn.parameters()[1], xg});

        BatchNorm bn(4);
        auto xb = tin({3, 4});
        check_xfail("BatchNorm", "no backward implemented at all", [&]{ return bn.forward(xb); },
              {bn.parameters()[0], bn.parameters()[1], xb});

        Embedding emb(6, 4);
        auto ids = make_shared<Tensor>(vector<int>{1, 3});
        ids->data[0] = 1;
        ids->data[1] = 4;
        ids->data[2] = 1;                     // repeated id: backward must accumulate
        GradCheckOptions all;
        all.max_points = 0;                   // probe every element
        check("Embedding (repeated id)", [&]{ return emb.forward(ids); },
              {emb.weight}, all);

        PositionalEncoding pe(16, 4);
        auto xp = tin({3, 4});
        check("PositionalEncoding", [&]{ return pe.forward(xp); }, {xp});
    }

    printf("\n=== recurrent layers ===\n");
    {
        // input is [seq_len, input_size]
        RNN rnn(3, 4);
        auto xr = tin({5, 1, 3});
        auto pr = rnn.parameters();
        check_xfail("RNN", "output sequence detached; hidden state is fine", [&]{ return rnn.forward(xr); }, {pr[0], pr[1], pr[2], xr});

        LSTM lstm(3, 4);
        auto xl = tin({5, 1, 3});
        vector<TensorPtr> li = lstm.parameters();
        li.push_back(xl);
        check_xfail("LSTM", "output sequence detached; hidden state is fine", [&]{ return lstm.forward(xl); }, li);

        GRU gru(3, 4);
        auto xu = tin({5, 1, 3});
        vector<TensorPtr> ui = gru.parameters();
        ui.push_back(xu);
        check_xfail("GRU", "output sequence detached; hidden state is fine", [&]{ return gru.forward(xu); }, ui);
    }

    printf("\n=== vision layers ===\n");
    {
        Conv2D conv(1, 2, 3);
        auto xc = tin({1, 1, 5, 5});
        check("Conv2D", [&]{ return conv.forward(xc); },
              {conv.parameters()[0], conv.parameters()[1], xc});

        ConvTranspose2D ct(1, 2, 3);
        auto xt = tin({1, 1, 4, 4});
        check("ConvTranspose2D", [&]{ return ct.forward(xt); },
              {ct.parameters()[0], ct.parameters()[1], xt});

        MaxPool2D mp(2);
        auto xm = tin({1, 1, 4, 4});
        check("MaxPool2D", [&]{ return mp.forward(xm); }, {xm});

        AvgPool2D ap(2);
        auto xa = tin({1, 1, 4, 4});
        check("AvgPool2D", [&]{ return ap.forward(xa); }, {xa});

        ResidualBlock rb(2, 2, 1, 1);
        auto xb = tin({1, 2, 4, 4});
        auto prb = rb.parameters();
        check_xfail("ResidualBlock", "contains GroupNorm, which has no backward", [&]{ return rb.forward(xb); },
              {prb[0], prb[prb.size() - 1], xb});
    }

    printf("\n=== attention / transformer ===\n");
    {
        MultiHeadAttention mha(4, 2, true);
        auto x = tin({3, 4});
        check("MultiHeadAttention", [&]{ return mha.forward(x); },
              {mha.W_q, mha.W_k, mha.W_v, mha.W_o, x});

        MultiHeadAttention rope(4, 2, true, true);
        auto xr = tin({3, 4});
        check("MultiHeadAttention (RoPE)", [&]{ return rope.forward(xr); },
              {rope.W_q, rope.W_k, rope.W_v, rope.W_o, xr});

        TransformerBlock blk(4, 2, 8, true);
        auto xb = tin({3, 4});
        check("TransformerBlock", [&]{ return blk.forward(xb); },
              {blk.attn.W_o, blk.ff2.weights, xb});
    }

    return test_report();
}
