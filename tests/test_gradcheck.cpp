// Numeric gradient check: verify every analytical backward pass against
// central finite differences.
//
//     d loss / d p  ~=  (loss(p + eps) - loss(p - eps)) / (2 * eps)
//
// This is the core correctness test for a from-scratch autograd library. A
// severed graph shows up as an analytical gradient of exactly 0 where the
// numeric one is non-zero -- which is precisely the PositionalEncoding bug.
//
// Everything is float32, so eps is a compromise: too small and the subtraction
// is swamped by rounding, too large and the quadratic term of the Taylor
// expansion shows up. 1e-2 with a 2% relative tolerance is the usable band.

#include "ml/tensor.hpp"
#include "ml/ops.hpp"
#include "ml/nn/linear.hpp"
#include "ml/nn/nlp/layernorm.hpp"
#include "ml/nn/nlp/embedding.hpp"
#include "ml/nn/nlp/attention.hpp"
#include "ml/nn/nlp/positional_encoding.hpp"
#include "ml/nn/nlp/transformer.hpp"
#include "ml/nn/vision/conv2d.hpp"
#include "ml/utils/testing.hpp"

#include <functional>
#include <vector>
#include <string>

using namespace std;

static const float EPS     = 1e-2f;
static const float REL_TOL = 2e-2f;   // 2%

// Scalarise any output so there is a single loss to differentiate.
static float loss_of(TensorPtr out) { return mean(out, -1)->data[0]; }

// Check d(mean(f()))/d param against finite differences, at a few indices.
static void gradcheck(const string& name,
                      TensorPtr param,
                      function<TensorPtr()> f,
                      int max_indices = 4) {
    // analytical
    param->reset_grad();
    auto out  = f();
    auto loss = mean(out, -1);
    loss->backward();
    vector<float> analytic = param->grad;

    int n     = param->num_el();
    int step  = (n + max_indices - 1) / max_indices;
    if (step < 1) step = 1;

    for (int i = 0; i < n; i += step) {
        const float orig = param->data[i];

        param->data[i] = orig + EPS;
        const float lp = loss_of(f());

        param->data[i] = orig - EPS;
        const float lm = loss_of(f());

        param->data[i] = orig;

        const float numeric = (lp - lm) / (2.0f * EPS);
        CHECK_REL((name + " grad[" + to_string(i) + "]").c_str(),
                  analytic[i], numeric, REL_TOL);
    }
}

static TensorPtr input2d(int rows, int cols, bool grad) {
    auto t = make_shared<Tensor>(vector<int>{rows, cols});
    for (int i = 0; i < t->num_el(); i++)
        t->data[i] = 0.1f * ((i % 7) - 3);          // deterministic, mixed signs
    t->requires_grad = grad;
    if (grad) t->grad.assign(t->num_el(), 0.0f);
    return t;
}

int main() {
    Tensor::manual_seed(1234);                      // reproducible weight init

    printf("=== ops ===\n");
    {
        auto x = input2d(3, 4, true);
        gradcheck("gelu wrt input",      x, [&]{ return gelu(x); });
        gradcheck("sigmoid wrt input",   x, [&]{ return sigmoid(x); });
        gradcheck("tanh wrt input",      x, [&]{ return tanh_op(x); });
        gradcheck("transpose_op wrt in", x, [&]{ return transpose_op(x); });
        gradcheck("reshape_op wrt in",   x, [&]{ return reshape_op(x, {4, 3}); });
        gradcheck("softmax wrt input",   x, [&]{ return softmax(x); });
    }

    printf("\n=== Linear ===\n");
    {
        Linear fc(4, 3);
        auto x = input2d(2, 4, true);
        gradcheck("Linear weights", fc.weights, [&]{ return fc.forward(x); });
        gradcheck("Linear bias",    fc.bias,    [&]{ return fc.forward(x); });
        gradcheck("Linear input",   x,          [&]{ return fc.forward(x); });
    }

    printf("\n=== LayerNorm ===\n");
    {
        LayerNorm ln(4);
        auto x = input2d(3, 4, true);
        gradcheck("LayerNorm gamma", ln.parameters()[0], [&]{ return ln.forward(x); });
        gradcheck("LayerNorm beta",  ln.parameters()[1], [&]{ return ln.forward(x); });
        gradcheck("LayerNorm input", x,                  [&]{ return ln.forward(x); });
    }

    printf("\n=== Embedding ===\n");
    {
        Embedding emb(6, 4);
        auto ids = make_shared<Tensor>(vector<int>{1, 3});
        ids->data[0] = 1; ids->data[1] = 4; ids->data[2] = 1;   // repeat id 1 on purpose
        gradcheck("Embedding weight (repeated id)", emb.weight,
                  [&]{ return emb.forward(ids); }, 6);
    }

    printf("\n=== PositionalEncoding (regression: graph must not be severed) ===\n");
    {
        PositionalEncoding pe(16, 4);
        auto x = input2d(3, 4, true);
        gradcheck("PositionalEncoding input", x, [&]{ return pe.forward(x); });
    }

    printf("\n=== MultiHeadAttention ===\n");
    {
        MultiHeadAttention mha(4, 2, /*causal=*/true);
        auto x = input2d(3, 4, true);
        gradcheck("MHA W_q",   mha.W_q, [&]{ return mha.forward(x); }, 3);
        gradcheck("MHA W_k",   mha.W_k, [&]{ return mha.forward(x); }, 3);
        gradcheck("MHA W_v",   mha.W_v, [&]{ return mha.forward(x); }, 3);
        gradcheck("MHA W_o",   mha.W_o, [&]{ return mha.forward(x); }, 3);
        gradcheck("MHA input", x,       [&]{ return mha.forward(x); }, 3);
    }

    printf("\n=== TransformerBlock ===\n");
    {
        TransformerBlock blk(4, 2, 8, /*causal=*/true);
        auto x = input2d(3, 4, true);
        gradcheck("TransformerBlock input", x, [&]{ return blk.forward(x); }, 3);
        gradcheck("TransformerBlock ff2 w", blk.ff2.weights,
                  [&]{ return blk.forward(x); }, 3);
    }

    printf("\n=== Conv2D ===\n");
    {
        Conv2D conv(1, 2, 3);                        // in=1, out=2, k=3
        auto x = make_shared<Tensor>(vector<int>{1, 1, 5, 5});
        for (int i = 0; i < x->num_el(); i++) x->data[i] = 0.1f * ((i % 5) - 2);
        x->requires_grad = true;
        x->grad.assign(x->num_el(), 0.0f);
        gradcheck("Conv2D kernel", conv.parameters()[0], [&]{ return conv.forward(x); }, 4);
        gradcheck("Conv2D input",  x,                    [&]{ return conv.forward(x); }, 4);
    }

    return test_report();
}
