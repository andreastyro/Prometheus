#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ml/tensor.hpp"
#include "ml/ops.hpp"
#include "ml/loss.hpp"
#include "ml/metrics/metrics.hpp"

#include "ml/nn/linear.hpp"
#include "ml/nn/activations.hpp"
#include "ml/nn/dropout.hpp"
#include "ml/nn/sequential.hpp"
#include "ml/nn/flatten.hpp"
#include "ml/nn/vision/conv2d.hpp"
#include "ml/nn/vision/residual_block.hpp"
#include "ml/nn/vision/maxpool2d.hpp"
#include "ml/nn/vision/avgpool2d.hpp"
#include "ml/nn/vision/convtranspose2d.hpp"
#include "ml/nn/rnn/rnn.hpp"
#include "ml/nn/rnn/lstm.hpp"
#include "ml/nn/rnn/gru.hpp"
#include "ml/nn/nlp/embedding.hpp"
#include "ml/nn/nlp/layernorm.hpp"
#include "ml/nn/groupnorm.hpp"
#include "ml/nn/nlp/attention.hpp"
#include "ml/nn/nlp/transformer.hpp"
#include "ml/nn/nlp/positional_encoding.hpp"
#include "ml/nn/nlp/gpt.hpp"

#include "ml/optim/sgd.hpp"
#include "ml/optim/adam.hpp"
#include "ml/optim/adamw.hpp"
#include "ml/optim/rmsprop.hpp"
#include "ml/optim/scheduler.hpp"
#include "ml/optim/grad_scaler.hpp"
#include "ml/utils/grad_clip.hpp"
#include "ml/utils/model_io.hpp"

namespace py = pybind11;

// Python-safe Sequential that keeps layer objects alive
struct PySequential {
    py::list held;
    std::shared_ptr<Sequential> impl;

    PySequential(py::list layers) : held(layers) {
        std::vector<Module*> raw;
        for (auto h : held)
            raw.push_back(h.cast<std::shared_ptr<Module>>().get());
        impl = std::make_shared<Sequential>(raw);
    }

    TensorPtr forward(TensorPtr x) { return impl->forward(x); }
    std::vector<TensorPtr> parameters() { return impl->parameters(); }
};

PYBIND11_MODULE(prometheus, m) {
    m.doc() = "Prometheus ML library";

    // -------------------------------------------------------------------------
    // Tensor
    // -------------------------------------------------------------------------
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>>())
        .def(py::init<std::vector<int>, std::vector<float>>())
        .def_readwrite("data",          &Tensor::data)
        .def_readwrite("shape",         &Tensor::shape)
        .def_readwrite("grad",          &Tensor::grad)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def("get",       &Tensor::get)
        .def("set",       &Tensor::set)
        .def("fill",      &Tensor::fill)
        .def("print",     &Tensor::print)
        .def("num_el",    &Tensor::num_el)
        .def("transpose", &Tensor::transpose)
        .def("reshape",   &Tensor::reshape)
        .def("backward",  &Tensor::backward)
        .def("reset_grad",&Tensor::reset_grad)
        .def("detach",    &Tensor::detach)
        .def("half",      &Tensor::half)
        .def("to_float",  &Tensor::to_float)
        .def_static("zeros", &Tensor::zeros)
        .def_static("ones",  &Tensor::ones)
        .def_static("randn", &Tensor::randn)
        // numpy interop
        .def("numpy", [](TensorPtr t) {
            std::vector<py::ssize_t> shape(t->shape.begin(), t->shape.end());
            return py::array_t<float>(shape, t->data.data());
        })
        .def_static("from_numpy", [](py::array_t<float> arr, std::vector<int> shape) {
            auto buf = arr.request();
            float* ptr = static_cast<float*>(buf.ptr);
            std::vector<float> data(ptr, ptr + buf.size);
            return std::make_shared<Tensor>(shape, data);
        });

    // -------------------------------------------------------------------------
    // Ops
    // -------------------------------------------------------------------------
    m.def("add",           py::overload_cast<TensorPtr, TensorPtr>(&add));
    m.def("add",           py::overload_cast<float, TensorPtr>(&add));
    m.def("subtract",      py::overload_cast<TensorPtr, TensorPtr>(&subtract));
    m.def("subtract",      py::overload_cast<float, TensorPtr>(&subtract));
    m.def("subtract",      py::overload_cast<TensorPtr, float>(&subtract));
    m.def("multiply",      py::overload_cast<TensorPtr, TensorPtr>(&multiply));
    m.def("multiply",      py::overload_cast<float, TensorPtr>(&multiply));
    m.def("divide",        py::overload_cast<TensorPtr, TensorPtr>(&divide));
    m.def("divide",        py::overload_cast<float, TensorPtr>(&divide));
    m.def("divide",        py::overload_cast<TensorPtr, float>(&divide));
    m.def("matmul",        &matmul);
    m.def("relu",          &relu);
    m.def("gelu",          &gelu);
    m.def("sigmoid",       &sigmoid);
    m.def("tanh_op",       &tanh_op);
    m.def("softmax",       &softmax);
    m.def("reshape_op",    &reshape_op, py::arg("a"), py::arg("new_shape"));
    m.def("log_op",        &log_op);
    m.def("exp_op",        &exp_op);
    m.def("pow_op",        &pow_op);
    m.def("sqrt_op",       &sqrt_op);
    m.def("abs_op",        &abs_op);
    m.def("sum",           &sum,     py::arg("a"), py::arg("axis") = -1);
    m.def("mean",          &mean,    py::arg("a"), py::arg("axis") = -1);
    m.def("max_op",        &max_op);
    m.def("min_op",        &min_op);
    m.def("clip",           &clip);
    m.def("broadcast_add", &broadcast_add);
    m.def("argmax",         &argmax);
    m.def("clip_grad_norm", &clip_grad_norm,
          py::arg("params"), py::arg("max_norm"));

    // -------------------------------------------------------------------------
    // Loss
    // -------------------------------------------------------------------------
    m.def("mse_loss",                &mse_loss);
    m.def("mae_loss",                &mae_loss);
    m.def("bce_loss",                &bce_loss);
    m.def("cross_entropy_loss",      &cross_entropy_loss);
    m.def("cross_entropy_sparse",    &cross_entropy_sparse,
          py::arg("logits"), py::arg("target_idx"));
    m.def("cross_entropy_sparse_seq",&cross_entropy_sparse_seq,
          py::arg("logits"), py::arg("targets"));
    m.def("huber_loss",         &huber_loss,         py::arg("pred"), py::arg("target"), py::arg("delta") = 1.0f);
    m.def("kl_divergence",      &kl_divergence);
    m.def("reconstruction_loss",&reconstruction_loss);
    m.def("contrastive_loss",   &contrastive_loss,   py::arg("anchor"), py::arg("positive"), py::arg("negative"), py::arg("margin") = 1.0f);
    m.def("l1_regularization",  &l1_regularization,  py::arg("params"), py::arg("lambda_"));
    m.def("l2_regularization",  &l2_regularization,  py::arg("params"), py::arg("lambda_"));

    // -------------------------------------------------------------------------
    // Metrics
    // -------------------------------------------------------------------------
    m.def("accuracy",         &accuracy);
    m.def("precision",        &precision);
    m.def("recall",           &recall);
    m.def("f1_score",         &f1_score);
    m.def("confusion_matrix", &confusion_matrix);
    m.def("r2_score",         &r2_score);

    // -------------------------------------------------------------------------
    // Module base (not instantiable, just for type checks)
    // -------------------------------------------------------------------------
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("forward",    &Module::forward)
        .def("parameters", &Module::parameters);

    // -------------------------------------------------------------------------
    // Layers
    // -------------------------------------------------------------------------
    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int, std::string>(),
             py::arg("in_features"), py::arg("out_features"), py::arg("init") = "default")
        .def("forward",    &Linear::forward)
        .def("parameters", &Linear::parameters)
        .def_readwrite("weights", &Linear::weights)
        .def_readwrite("bias",    &Linear::bias);

    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>())
        .def("forward",    &ReLU::forward)
        .def("parameters", &ReLU::parameters);

    py::class_<Sigmoid, Module, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward",    &Sigmoid::forward)
        .def("parameters", &Sigmoid::parameters);

    py::class_<Tanh, Module, std::shared_ptr<Tanh>>(m, "Tanh")
        .def(py::init<>())
        .def("forward",    &Tanh::forward)
        .def("parameters", &Tanh::parameters);

    py::class_<Softmax, Module, std::shared_ptr<Softmax>>(m, "Softmax")
        .def(py::init<>())
        .def("forward",    &Softmax::forward)
        .def("parameters", &Softmax::parameters);

    py::class_<Dropout, Module, std::shared_ptr<Dropout>>(m, "Dropout")
        .def(py::init<float, bool>(), py::arg("rate"), py::arg("training") = true)
        .def("forward",    &Dropout::forward)
        .def("parameters", &Dropout::parameters)
        .def_readwrite("training", &Dropout::training);

    py::class_<Flatten, Module, std::shared_ptr<Flatten>>(m, "Flatten")
        .def(py::init<>())
        .def("forward",    &Flatten::forward)
        .def("parameters", &Flatten::parameters);

    py::class_<Conv2D, Module, std::shared_ptr<Conv2D>>(m, "Conv2D")
        .def(py::init<int, int, int, int, int, std::string>(),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
             py::arg("stride") = 1, py::arg("padding") = 0, py::arg("weight_init") = "default")
        .def("forward",    &Conv2D::forward)
        .def("parameters", &Conv2D::parameters);

    py::class_<ResidualBlock, Module, std::shared_ptr<ResidualBlock>>(m, "ResidualBlock")
        .def(py::init<int, int, int, int>(),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("stride") = 1, py::arg("num_groups") = 1)
        .def("forward",    &ResidualBlock::forward)
        .def("parameters", &ResidualBlock::parameters);

    py::class_<MaxPool2D, Module, std::shared_ptr<MaxPool2D>>(m, "MaxPool2D")
        .def(py::init<int, int>(), py::arg("kernel_size"), py::arg("stride") = -1)
        .def("forward",    &MaxPool2D::forward)
        .def("parameters", &MaxPool2D::parameters);

    py::class_<AvgPool2D, Module, std::shared_ptr<AvgPool2D>>(m, "AvgPool2D")
        .def(py::init<int, int>(), py::arg("kernel_size"), py::arg("stride") = -1)
        .def("forward",    &AvgPool2D::forward)
        .def("parameters", &AvgPool2D::parameters);

    py::class_<ConvTranspose2D, Module, std::shared_ptr<ConvTranspose2D>>(m, "ConvTranspose2D")
        .def(py::init<int, int, int, int, int, std::string>(),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
             py::arg("stride") = 1, py::arg("padding") = 0, py::arg("weight_init") = "default")
        .def("forward",    &ConvTranspose2D::forward)
        .def("parameters", &ConvTranspose2D::parameters);

    py::class_<RNN, Module, std::shared_ptr<RNN>>(m, "RNN")
        .def(py::init<int, int>())
        .def("forward",    py::overload_cast<TensorPtr>(&RNN::forward))
        .def("forward_with_state", [](RNN& r, TensorPtr x, TensorPtr h0) {
            return r.forward(x, h0);
        })
        .def("parameters", &RNN::parameters);

    py::class_<LSTM, Module, std::shared_ptr<LSTM>>(m, "LSTM")
        .def(py::init<int, int>())
        .def("forward",    py::overload_cast<TensorPtr>(&LSTM::forward))
        .def("forward_with_state", [](LSTM& l, TensorPtr x, TensorPtr h0, TensorPtr c0) {
            return l.forward(x, h0, c0);
        })
        .def("parameters", &LSTM::parameters);

    py::class_<GRU, Module, std::shared_ptr<GRU>>(m, "GRU")
        .def(py::init<int, int>())
        .def("forward",    py::overload_cast<TensorPtr>(&GRU::forward))
        .def("forward_with_state", [](GRU& g, TensorPtr x, TensorPtr h0) {
            return g.forward(x, h0);
        })
        .def("parameters", &GRU::parameters);

    py::class_<Embedding, Module, std::shared_ptr<Embedding>>(m, "Embedding")
        .def(py::init<int, int>())
        .def("forward",    &Embedding::forward)
        .def("parameters", &Embedding::parameters)
        .def_readwrite("weight", &Embedding::weight);

    py::class_<LayerNorm, Module, std::shared_ptr<LayerNorm>>(m, "LayerNorm")
        .def(py::init<int, float>(), py::arg("normalized_shape"), py::arg("eps") = 1e-5f)
        .def("forward",    &LayerNorm::forward)
        .def("parameters", &LayerNorm::parameters);

    py::class_<GroupNorm, Module, std::shared_ptr<GroupNorm>>(m, "GroupNorm")
        .def(py::init<int, int, float>(),
             py::arg("num_groups"), py::arg("num_channels"), py::arg("eps") = 1e-5f)
        .def("forward",    &GroupNorm::forward)
        .def("parameters", &GroupNorm::parameters);

    py::class_<MultiHeadAttention, Module, std::shared_ptr<MultiHeadAttention>>(m, "MultiHeadAttention")
        .def(py::init<int, int, bool, bool>(),
             py::arg("embed_dim"), py::arg("num_heads"),
             py::arg("causal") = false, py::arg("rope") = false)
        .def("forward",    &MultiHeadAttention::forward)
        .def("parameters", &MultiHeadAttention::parameters);

    py::class_<TransformerBlock, Module, std::shared_ptr<TransformerBlock>>(m, "TransformerBlock")
        .def(py::init<int, int, int, bool, bool>(),
             py::arg("embed_dim"), py::arg("num_heads"), py::arg("ff_dim"),
             py::arg("causal") = false, py::arg("rope") = false)
        .def("forward",    &TransformerBlock::forward)
        .def("parameters", &TransformerBlock::parameters);

    py::class_<KVCache>(m, "KVCache")
        .def("reset",    &KVCache::reset)
        .def_readonly("past_len", &KVCache::past_len);

    py::class_<GPT>(m, "GPT")
        .def(py::init<int, int, int, int, int, int, bool>(),
             py::arg("vocab_size"), py::arg("max_seq_len"), py::arg("embed_dim"),
             py::arg("num_heads"), py::arg("num_layers"),
             py::arg("ff_dim") = 0, py::arg("rope") = false)
        .def("forward",         &GPT::forward)
        .def("forward_cached",  &GPT::forward_cached,
             py::arg("token_id"), py::arg("cache"))
        .def("make_kv_cache",   &GPT::make_kv_cache)
        .def("parameters",      &GPT::parameters)
        .def_readonly("vocab_size_",   &GPT::vocab_size_)
        .def_readonly("embed_dim_",    &GPT::embed_dim_)
        .def_readonly("max_seq_len_",  &GPT::max_seq_len_)
        .def_readonly("num_layers_",   &GPT::num_layers_)
        .def_readonly("rope_",         &GPT::rope_);

    py::class_<PositionalEncoding>(m, "PositionalEncoding")
        .def(py::init<int, int>(), py::arg("max_len"), py::arg("embed_dim"))
        .def("forward", &PositionalEncoding::forward);

    // Sequential — PySequential wrapper keeps layer objects alive
    py::class_<PySequential>(m, "Sequential")
        .def(py::init<py::list>())
        .def("forward",    &PySequential::forward)
        .def("parameters", &PySequential::parameters);

    // -------------------------------------------------------------------------
    // Optimizers
    // -------------------------------------------------------------------------
    py::class_<SGD>(m, "SGD")
        .def(py::init<std::vector<TensorPtr>, float, float>(),
             py::arg("params"), py::arg("lr"), py::arg("momentum") = 0.0f)
        .def("step",       &SGD::step)
        .def("zero_grad",  &SGD::zero_grad)
        .def_readwrite("lr", &SGD::lr);

    py::class_<Adam>(m, "Adam")
        .def(py::init<std::vector<TensorPtr>, float, float, float, float>(),
             py::arg("params"), py::arg("lr") = 0.001f,
             py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
        .def("step",       &Adam::step)
        .def("zero_grad",  &Adam::zero_grad)
        .def("save_state", &Adam::save_state, py::arg("path"))
        .def("load_state", &Adam::load_state, py::arg("path"))
        .def_readwrite("lr", &Adam::lr);

    py::class_<AdamW>(m, "AdamW")
        .def(py::init<std::vector<TensorPtr>, float, float, float, float, float>(),
             py::arg("params"), py::arg("lr") = 0.001f,
             py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f,
             py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.01f)
        .def("step",         &AdamW::step)
        .def("zero_grad",    &AdamW::zero_grad)
        .def("save_state",   &AdamW::save_state, py::arg("path"))
        .def("load_state",   &AdamW::load_state, py::arg("path"))
        .def_readwrite("lr",           &AdamW::lr)
        .def_readwrite("weight_decay", &AdamW::weight_decay);

    py::class_<RMSprop>(m, "RMSprop")
        .def(py::init<std::vector<TensorPtr>, float, float, float>(),
             py::arg("params"), py::arg("lr") = 0.001f,
             py::arg("beta") = 0.9f, py::arg("eps") = 1e-8f)
        .def("step",      &RMSprop::step)
        .def("zero_grad", &RMSprop::zero_grad)
        .def_readwrite("lr", &RMSprop::lr);

    // -------------------------------------------------------------------------
    // LR Schedulers
    // -------------------------------------------------------------------------
    py::class_<LRScheduler>(m, "LRScheduler")
        .def("step",   &LRScheduler::step)
        .def("get_lr", &LRScheduler::get_lr);

    py::class_<StepLR, LRScheduler>(m, "StepLR")
        .def(py::init<float, int, float>(),
             py::arg("base_lr"), py::arg("step_size"), py::arg("gamma") = 0.1f)
        .def("step",   &StepLR::step)
        .def("get_lr", &StepLR::get_lr);

    py::class_<ExponentialLR, LRScheduler>(m, "ExponentialLR")
        .def(py::init<float, float>(),
             py::arg("base_lr"), py::arg("gamma") = 0.95f)
        .def("step",   &ExponentialLR::step)
        .def("get_lr", &ExponentialLR::get_lr);

    py::class_<CosineAnnealingLR, LRScheduler>(m, "CosineAnnealingLR")
        .def(py::init<float, int, float>(),
             py::arg("base_lr"), py::arg("T_max"), py::arg("min_lr") = 0.0f)
        .def("step",   &CosineAnnealingLR::step)
        .def("get_lr", &CosineAnnealingLR::get_lr);

    // -------------------------------------------------------------------------
    // GradScaler (mixed precision)
    // -------------------------------------------------------------------------
    py::class_<GradScaler>(m, "GradScaler")
        .def(py::init<float, float, float, int, bool>(),
             py::arg("init_scale")      = 65536.0f,
             py::arg("growth_factor")   = 2.0f,
             py::arg("backoff_factor")  = 0.5f,
             py::arg("growth_interval") = 2000,
             py::arg("enabled")         = true)
        .def("scale_loss", &GradScaler::scale_loss)
        .def("unscale",    &GradScaler::unscale)
        .def("update",     &GradScaler::update)
        .def_readwrite("scale",   &GradScaler::scale)
        .def_readwrite("enabled", &GradScaler::enabled);

    // -------------------------------------------------------------------------
    // Model I/O
    // -------------------------------------------------------------------------
    m.def("save_model", &save, py::arg("path"), py::arg("params"));
    m.def("load_model", &load, py::arg("path"));

    py::class_<Checkpoint>(m, "Checkpoint")
        .def_readonly("epoch", &Checkpoint::epoch)
        .def_readonly("loss",  &Checkpoint::loss);

    m.def("save_checkpoint", &save_checkpoint,
          py::arg("path"), py::arg("params"), py::arg("epoch"), py::arg("loss"));
    // Returns (Checkpoint, list[Tensor]) — Python can't mutate a passed-in list.
    m.def("load_checkpoint", [](const std::string& path) {
        std::vector<TensorPtr> params;
        Checkpoint ckpt = load_checkpoint(path, params);
        return py::make_tuple(ckpt, params);
    }, py::arg("path"));

}
