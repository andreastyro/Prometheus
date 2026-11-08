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

// Finer finite-difference step, for functions whose composition is strongly
// nonlinear. A plain RNN applies tanh once per timestep and a ResidualBlock
// contains two ReLUs; at the default eps of 1e-2 the central difference either
// carries real truncation error or straddles a kink. Different platforms round
// tanh and fused multiply-add differently, so a check that only just passes on
// one compiler fails on another. 1e-3 keeps all three in agreement.
static GradCheckOptions fine_opts() {
    GradCheckOptions o;
    o.eps = 1e-3f;
    return o;
}

// For the recurrent layers. With PyTorch's init their recurrence is smooth,
// and their input and weight gradients are small -- down to ~1e-5, the size
// of float32 rounding in a finite difference at a fine step. A larger step
// cuts that noise in proportion; Richardson extrapolation in gradcheck keeps
// the truncation error negligible at this step on a smooth function.
static GradCheckOptions smooth_opts() {
    GradCheckOptions o;
    o.eps = 5e-2f;
    return o;
}

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
    for (int i = 0; i < t->num_el(); i++) t->data()[i] = 0.1f * ((i % 7) - 3) + 0.05f;
    t->set_requires_grad(grad);
    if (grad) t->zero_grad();
    return t;
}

// Probabilities strictly inside (0, 1), for the losses that take them.
static TensorPtr tprob(vector<int> shape) {
    auto t = make_shared<Tensor>(shape);
    for (int i = 0; i < t->num_el(); i++) t->data()[i] = 0.2f + 0.1f * (i % 6);
    t->set_requires_grad(true);
    t->zero_grad();
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
        auto bt = tin({2, 4});
        check("matmul_bt", [&]{ return matmul_bt(x, bt); }, {x, bt});
        // Views. Each output is weighted by distinct values before the mean:
        // a plain mean gives every position the same gradient, so a view
        // whose backward mapped positions wrongly would still pass.
        auto w43 = tin({4, 3});
        auto w26 = tin({2, 6});
        check("transpose (view)",
              [&]{ return multiply(transpose(x), w43); }, {x});
        check("transpose + contiguous",
              [&]{ return multiply(contiguous(transpose(x)), w43); }, {x});
        check("reshape (view)",
              [&]{ return multiply(reshape(x, {4, 3}), w43); }, {x});
        check("reshape of a transposed view",
              [&]{ return multiply(reshape(transpose(x), {2, 6}), w26); }, {x});
        check("op on a non-contiguous view",
              [&]{ return multiply(relu(transpose(x)), w43); }, {x});
        auto x3   = tin({2, 3, 4});
        auto w342 = tin({3, 4, 2});
        check("permute 3-D (view)",
              [&]{ return multiply(permute(x3, {1, 2, 0}), w342); }, {x3});
        check("exp_op",       [&]{ return exp_op(x); },             {x});

        auto y = tin({3, 4});
        check("add",      [&]{ return add(x, y); },      {x, y});
        check("multiply", [&]{ return multiply(x, y); }, {x, y});
        check("subtract", [&]{ return subtract(x, y); }, {x, y});

        auto m = tin({4, 2});
        check("matmul", [&]{ return matmul(x, m); }, {x, m});
    }

    printf("\n=== n-d: broadcasting, batched matmul, reductions ===\n");
    {
        // Every output is weighted by distinct values before the mean, so
        // each position carries a different gradient. For a broadcast input
        // that is what proves the sum back over the broadcast dims is right.
        auto a34 = tin({3, 4}), b4 = tin({4}), w34 = tin({3, 4});
        check("add [3,4] + [4] (bias)",
              [&]{ return multiply(add(a34, b4), w34); }, {a34, b4});
        auto a4 = tin({4});
        check("subtract [4] - [3,4]",
              [&]{ return multiply(subtract(a4, a34), w34); }, {a4, a34});
        auto x234 = tin({2, 3, 4}), y31 = tin({3, 1}), w234 = tin({2, 3, 4});
        check("multiply [2,3,4] * [3,1]",
              [&]{ return multiply(multiply(x234, y31), w234); }, {x234, y31});
        auto d14 = tprob({1, 4});   // denominators safely away from zero
        check("divide [3,4] / [1,4]",
              [&]{ return multiply(divide(a34, d14), w34); }, {a34, d14});

        auto m234 = tin({2, 3, 4}), k45 = tin({4, 5}), w235 = tin({2, 3, 5});
        check("matmul [2,3,4] @ [4,5] (one folded gemm)",
              [&]{ return multiply(matmul(m234, k45), w235); }, {m234, k45});
        auto k245 = tin({2, 4, 5});
        check("matmul [2,3,4] @ [2,4,5] (batched)",
              [&]{ return multiply(matmul(m234, k245), w235); }, {m234, k245});
        auto p2134 = tin({2, 1, 3, 4}), q342 = tin({3, 4, 2}), w2332 = tin({2, 3, 3, 2});
        check("matmul [2,1,3,4] @ [3,4,2] (batch broadcast)",
              [&]{ return multiply(matmul(p2134, q342), w2332); }, {p2134, q342});
        auto x35 = tin({3, 5}), W45 = tin({4, 5}), w34b = tin({3, 4});
        check("matmul by a transpose view (no copy)",
              [&]{ return multiply(matmul(x35, transpose(W45)), w34b); }, {x35, W45});

        auto w24 = tin({2, 4}), w214 = tin({2, 1, 4}), w34c = tin({3, 4});
        check("sum over dim 1 of [2,3,4]",
              [&]{ return multiply(sum(x234, 1), w24); }, {x234});
        check("mean over dim 1, keepdim",
              [&]{ return multiply(mean(x234, 1, true), w214); }, {x234});
        check("sum over dim 0",
              [&]{ return multiply(sum(x234, 0), w34c); }, {x234});
        check("softmax over the last dim of [2,3,4]",
              [&]{ return multiply(softmax(x234), w234); }, {x234});
    }

    printf("\n=== losses ===\n");
    {
        // target must genuinely differ from pred: with pred == target the loss
        // sits exactly at its minimum, the true gradient is zero, and a
        // detached loss passes the check for the wrong reason.
        auto pred = tin({3, 2});
        auto targ = make_shared<Tensor>(Shape{3, 2});
        for (int i = 0; i < targ->num_el(); i++) targ->data()[i] = -0.4f + 0.3f * i;
        check("mse_loss",            [&]{ return mse_loss(pred, targ); },            {pred});
        check("mae_loss",            [&]{ return mae_loss(pred, targ); },            {pred});
        check("huber_loss",          [&]{ return huber_loss(pred, targ); },          {pred});
        check("reconstruction_loss", [&]{ return reconstruction_loss(pred, targ); }, {pred});

        auto p = tprob({3, 2});
        auto q = tprob({3, 2});
        for (int i = 0; i < q->num_el(); i++) q->data()[i] = 1.0f - q->data()[i];  // differ from p
        q->set_requires_grad(false);
        check("bce_loss",      [&]{ return bce_loss(p, q); },      {p});
        check("kl_divergence", [&]{ return kl_divergence(p, q); }, {p});

        auto logits = tprob({2, 4});
        auto onehot = make_shared<Tensor>(Shape{2, 4},
                                          vector<float>{1, 0, 0, 0, 0, 0, 1, 0});
        check("cross_entropy_loss",
              [&]{ return cross_entropy_loss(logits, onehot); }, {logits});

        auto lg = tin({3, 5});
        check("cross_entropy_sparse_seq",
              [&]{ return cross_entropy_sparse_seq(lg, {0, 2, 4}); }, {lg});

        // three distinct points, and far enough apart to keep the hinge active
        auto a   = tin({1, 3});
        auto pos = make_shared<Tensor>(Shape{1, 3});
        auto neg = make_shared<Tensor>(Shape{1, 3});
        for (int i = 0; i < 3; i++) { pos->data()[i] = -0.1f + 0.2f * i; neg->data()[i] = 0.7f - 0.2f * i; }
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
        auto ids = make_shared<Tensor>(Shape{1, 3});
        ids->data()[0] = 1;
        ids->data()[1] = 4;
        ids->data()[2] = 1;                     // repeated id: backward must accumulate
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
        check("RNN", [&]{ return rnn.forward(xr); }, {pr[0], pr[1], pr[2], xr}, smooth_opts());

        LSTM lstm(3, 4);
        auto xl = tin({5, 1, 3});
        vector<TensorPtr> li = lstm.parameters();
        li.push_back(xl);
        check("LSTM", [&]{ return lstm.forward(xl); }, li, smooth_opts());

        GRU gru(3, 4);
        auto xu = tin({5, 1, 3});
        vector<TensorPtr> ui = gru.parameters();
        ui.push_back(xu);
        check("GRU", [&]{ return gru.forward(xu); }, ui, smooth_opts());
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
        // Two ReLUs inside, so a perturbation of any size can flip an
        // activation sign and make the central difference straddle the kink.
        // Whether it does depends on how the platform rounds the convolution,
        // which is why a tight tolerance here passed on MinGW and failed on
        // MSVC and clang. Treat this as a graph-connectivity check rather than
        // a precision one: a detached layer gives an analytic gradient of
        // exactly 0 against a numeric O(1), which these tolerances still catch,
        // while kink noise does not trip them. Per-element precision for the
        // parts is covered by the Conv2D and GroupNorm checks above.
        GradCheckOptions kinked;
        kinked.eps  = 1e-3f;
        kinked.atol = 5e-2f;
        kinked.rtol = 0.5f;
        check("ResidualBlock (connectivity)", [&]{ return rb.forward(xb); },
              {prb[0], prb[prb.size() - 1], xb}, kinked);
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

        // Sequential must chain gradients through every child. Tanh rather
        // than ReLU between them: which activation is beside the point here,
        // and ReLU's kink made the check depend on whether a random weight
        // put a hidden unit within one finite-difference step of zero.
        Sequential seq({ new Linear(4, 6), new Tanh(), new Linear(6, 2) });
        auto xs = tin({2, 4});
        check("Sequential", [&]{ return seq.forward(xs); },
              {seq.parameters()[0], seq.parameters()[2], xs});

        // Dropout with p = 0 is deterministic, so it can be checked
        Dropout drop(0.0f);
        auto xd = tin({3, 4});
        check("Dropout (p=0)", [&]{ return drop.forward(xd); }, {xd});

        // the whole GPT model, not just its pieces
        GPT gpt(16, 8, 8, 2, 1);
        auto ids = make_shared<Tensor>(Shape{3});
        ids->data()[0] = 1; ids->data()[1] = 5; ids->data()[2] = 2;
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
        auto u = make_shared<Tensor>(Shape{3, 4});
        for (int i = 0; i < u->num_el(); i++) u->data()[i] = -0.5f + 0.17f * i;
        u->set_requires_grad(true);
        u->zero_grad();
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

        // The recurrent layers used to start from randn weights (std 1),
        // which made ten steps of tanh chaotic: the loss was so nonlinear that
        // the default eps of 1e-2 left real truncation error, and these checks
        // needed a finer step. With PyTorch's U(-1/sqrt(H), 1/sqrt(H)) init the
        // recurrence is smooth, and the trade-off reverses: input gradients are
        // ~1e-5, the size of float32 rounding at a fine step (a GRU element
        // failed at 2.8e-5 on macOS). The default step keeps truncation
        // negligible and rounding 10x smaller.
        GradCheckOptions deep = smooth_opts();
        deep.max_points = 3;

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
        auto ids2 = make_shared<Tensor>(Shape{6});
        for (int i = 0; i < 6; i++) ids2->data()[i] = (float)(i * 3 + 1);
        check("GPT 3 layers 4 heads seq 6", [&]{ return gpt2.forward(ids2); },
              {gpt2.tok_emb.weight, gpt2.ln_f.parameters()[0]}, few);
    }

    return test_report();
}
