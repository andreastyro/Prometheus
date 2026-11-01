// Numeric gradient check over every layer, op and loss in the library,
// driven by the public gradcheck() API.

#include "ml/tensor.hpp"
#include "ml/ops.hpp"
#include "ml/loss.hpp"
#include "ml/optim/grad_scaler.hpp"
#include "ml/nn/linear.hpp"
#include "ml/nn/nlp/gpt.hpp"
#include "ml/nn/dropout.hpp"
#include "ml/nn/activations.hpp"
#include "ml/nn/sequential.hpp"
#include "ml/nn/flatten.hpp"
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
        // target must genuinely differ from pred: with pred == target the loss
        // sits exactly at its minimum, the true gradient is zero, and a
        // detached loss passes the check for the wrong reason.
        auto pred = tin({3, 2});
        auto targ = make_shared<Tensor>(vector<int>{3, 2});
        for (int i = 0; i < targ->num_el(); i++) targ->data[i] = -0.4f + 0.3f * i;
        check("mse_loss",            [&]{ return mse_loss(pred, targ); },            {pred});
        check("mae_loss",            [&]{ return mae_loss(pred, targ); },            {pred});
        check("huber_loss",          [&]{ return huber_loss(pred, targ); },          {pred});
        check("reconstruction_loss", [&]{ return reconstruction_loss(pred, targ); }, {pred});

        auto p = tprob({3, 2});
        auto q = tprob({3, 2});
        for (int i = 0; i < q->num_el(); i++) q->data[i] = 1.0f - q->data[i];  // differ from p
        q->requires_grad = false;
        check("bce_loss",      [&]{ return bce_loss(p, q); },      {p});
        check("kl_divergence", [&]{ return kl_divergence(p, q); }, {p});

        auto logits = tprob({2, 4});
        auto onehot = make_shared<Tensor>(vector<int>{2, 4},
                                          vector<float>{1, 0, 0, 0, 0, 0, 1, 0});
        check("cross_entropy_loss",
              [&]{ return cross_entropy_loss(logits, onehot); }, {logits});

        auto lg = tin({3, 5});
        check("cross_entropy_sparse_seq",
              [&]{ return cross_entropy_sparse_seq(lg, {0, 2, 4}); }, {lg});

        // three distinct points, and far enough apart to keep the hinge active
        auto a   = tin({1, 3});
        auto pos = make_shared<Tensor>(vector<int>{1, 3});
        auto neg = make_shared<Tensor>(vector<int>{1, 3});
        for (int i = 0; i < 3; i++) { pos->data[i] = -0.1f + 0.2f * i; neg->data[i] = 0.7f - 0.2f * i; }
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
        check("GroupNorm", [&]{ return gn.forward(xg); },
              {gn.parameters()[0], gn.parameters()[1], xg});

        BatchNorm bn(4);
        auto xb = tin({3, 4});
        check("BatchNorm", [&]{ return bn.forward(xb); },
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
        check("RNN", [&]{ return rnn.forward(xr); }, {pr[0], pr[1], pr[2], xr});

        LSTM lstm(3, 4);
        auto xl = tin({5, 1, 3});
        vector<TensorPtr> li = lstm.parameters();
        li.push_back(xl);
        check("LSTM", [&]{ return lstm.forward(xl); }, li);

        GRU gru(3, 4);
        auto xu = tin({5, 1, 3});
        vector<TensorPtr> ui = gru.parameters();
        ui.push_back(xu);
        check("GRU", [&]{ return gru.forward(xu); }, ui);
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
        // Two ReLUs inside: the default eps = 1e-2 is large enough to flip an
        // activation sign, which makes the central difference straddle the kink
        // and report a false mismatch. A smaller step stays on one side.
        GradCheckOptions relu_safe;
        relu_safe.eps = 1e-3f;
        check("ResidualBlock", [&]{ return rb.forward(xb); },
              {prb[0], prb[prb.size() - 1], xb}, relu_safe);
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

    printf("\n=== containers, activations and whole models ===\n");
    {
        // Flatten reinterprets the shape; the gradient must pass straight
        // through. When this was detached it silently froze every convolution
        // in the usual Conv2D -> Flatten -> Linear arrangement.
        Flatten fl;
        auto xf = tin({2, 1, 3, 3});
        check("Flatten", [&]{ return fl.forward(xf); }, {xf});

        // the realistic CNN path, end to end
        Conv2D cv(1, 2, 3);
        Flatten f2;
        Linear  head(2 * 3 * 3, 2);
        auto xi = tin({1, 1, 5, 5});
        check("Conv2D -> Flatten -> Linear",
              [&]{ return head.forward(f2.forward(cv.forward(xi))); },
              {cv.parameters()[0], head.weights, xi});

        // the Module wrappers around the activation ops
        ReLU relu_m; Sigmoid sig_m; Tanh tanh_m; Softmax soft_m;
        auto xa = tin({3, 4});
        check("ReLU (Module)",    [&]{ return relu_m.forward(xa); }, {xa});
        check("Sigmoid (Module)", [&]{ return sig_m.forward(xa);  }, {xa});
        check("Tanh (Module)",    [&]{ return tanh_m.forward(xa); }, {xa});
        check("Softmax (Module)", [&]{ return soft_m.forward(xa); }, {xa});

        // Sequential must chain gradients through every child
        Sequential seq({ new Linear(4, 6), new ReLU(), new Linear(6, 2) });
        auto xs = tin({2, 4});
        check("Sequential", [&]{ return seq.forward(xs); },
              {seq.parameters()[0], seq.parameters()[2], xs});

        // Dropout with p = 0 is deterministic, so it can be checked
        Dropout drop(0.0f);
        auto xd = tin({3, 4});
        check("Dropout (p=0)", [&]{ return drop.forward(xd); }, {xd});

        // the whole GPT model, not just its pieces
        GPT gpt(16, 8, 8, 2, 1);
        auto ids = make_shared<Tensor>(vector<int>{3});
        ids->data[0] = 1; ids->data[1] = 5; ids->data[2] = 2;
        GradCheckOptions few; few.max_points = 3;
        check("GPT (whole model)", [&]{ return gpt.forward(ids); },
              {gpt.tok_emb.weight, gpt.ln_f.parameters()[0]}, few);
    }

    printf("\n=== regularisers, reductions and the mixed-precision path ===\n");
    {
        auto w = tin({3, 3});
        check("l1_regularization", [&]{ return l1_regularization({w}, 0.1f); }, {w});
        check("l2_regularization", [&]{ return l2_regularization({w}, 0.1f); }, {w});

        // max_op and min_op are piecewise linear: the gradient goes entirely to
        // the element that attained the extremum. A tie makes the subgradient
        // ambiguous (the analytic form picks the first occurrence, the central
        // difference averages the two sides), so use strictly increasing data
        // where the extremum is unique.
        auto u = make_shared<Tensor>(vector<int>{3, 4});
        for (int i = 0; i < u->num_el(); i++) u->data[i] = -0.5f + 0.17f * i;
        u->requires_grad = true;
        u->grad.assign(u->num_el(), 0.0f);
        GradCheckOptions all_pts;
        all_pts.max_points = 0;
        check("max_op", [&]{ return max_op(u); }, {u}, all_pts);
        check("min_op", [&]{ return min_op(u); }, {u}, all_pts);

        // Scaling a loss for mixed precision must stay in the graph, otherwise
        // backward() reaches nothing and training silently stops.
        GradScaler gs(8.0f);
        auto l = tin({1});
        check("GradScaler::scale_loss",
              [&]{ return gs.scale_loss(mean(multiply(l, l), -1)); }, {l});
    }

    printf("\n=== wider shapes: batches, longer sequences, more heads ===\n");
    {
        GradCheckOptions few;
        few.max_points = 3;

        // A plain RNN composes tanh ten times, so the loss is strongly
        // nonlinear in each parameter and the central difference picks up real
        // truncation error at the default eps of 1e-2. The error falls as eps^2
        // (4.8e-3 -> 4.7e-4 -> 1.2e-5 at eps 1e-2, 3e-3, 1e-3), which is the
        // signature of a correct analytical gradient being compared against a
        // too-coarse difference, so use a finer step for the deep sequences.
        GradCheckOptions deep;
        deep.max_points = 3;
        deep.eps = 1e-3f;

        // recurrent layers with batch > 1 and a longer sequence, so the
        // backward-through-time path is exercised past a single step
        RNN rnn2(4, 6);
        auto xr2 = tin({10, 3, 4});
        auto pr2 = rnn2.parameters();
        check("RNN seq 10 batch 3", [&]{ return rnn2.forward(xr2); },
              {pr2[0], pr2[1], pr2[2], xr2}, deep);

        LSTM lstm2(4, 6);
        auto xl2 = tin({10, 3, 4});
        vector<TensorPtr> li2 = lstm2.parameters();
        li2.push_back(xl2);
        check("LSTM seq 10 batch 3", [&]{ return lstm2.forward(xl2); }, li2, deep);

        GRU gru2(4, 6);
        auto xu2 = tin({10, 3, 4});
        vector<TensorPtr> ui2 = gru2.parameters();
        ui2.push_back(xu2);
        check("GRU seq 10 batch 3", [&]{ return gru2.forward(xu2); }, ui2, deep);

        // Conv2D with a real batch, several channels, stride and padding
        Conv2D cv2(3, 4, 3, 2, 1);
        auto xc2 = tin({2, 3, 7, 7});
        check("Conv2D batch 2 stride 2 pad 1", [&]{ return cv2.forward(xc2); },
              {cv2.parameters()[0], cv2.parameters()[1], xc2}, few);

        // attention with more heads over a longer sequence
        MultiHeadAttention mha4(8, 4, true);
        auto xa4 = tin({7, 8});
        check("MultiHeadAttention 4 heads seq 7",
              [&]{ return mha4.forward(xa4); },
              {mha4.W_q, mha4.W_o, xa4}, few);

        // BatchNorm and GroupNorm with more than one batch item
        BatchNorm bn2(5);
        auto xb2 = tin({6, 5});
        check("BatchNorm batch 6", [&]{ return bn2.forward(xb2); },
              {bn2.parameters()[0], bn2.parameters()[1], xb2}, few);

        GroupNorm gn2(3, 6);
        auto xg2 = tin({2, 6, 3, 3});
        check("GroupNorm batch 2, 3 groups", [&]{ return gn2.forward(xg2); },
              {gn2.parameters()[0], gn2.parameters()[1], xg2}, few);

        // a deeper GPT than the single-layer smoke check
        GPT gpt2(24, 12, 16, 4, 3);
        auto ids2 = make_shared<Tensor>(vector<int>{6});
        for (int i = 0; i < 6; i++) ids2->data[i] = (float)(i * 3 + 1);
        check("GPT 3 layers 4 heads seq 6", [&]{ return gpt2.forward(ids2); },
              {gpt2.tok_emb.weight, gpt2.ln_f.parameters()[0]}, few);
    }

    return test_report();
}
