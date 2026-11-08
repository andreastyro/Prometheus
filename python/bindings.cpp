#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "ml/tensor.hpp"
#include "ml/strided.hpp"
#include "ml/ops.hpp"
#include "ml/matmul_backend.hpp"
#include "ml/blas.hpp"
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
#include "ml/nn/batchnorm.hpp"
#include "ml/utils/gradcheck.hpp"
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
    py::enum_<DType>(m, "dtype")
        .value("float32",  DType::Float32)
        .value("float64",  DType::Float64)
        .value("float16",  DType::Float16)
        .value("bfloat16", DType::BFloat16)
        .value("int64",    DType::Int64)
        .value("int32",    DType::Int32)
        .value("bool",     DType::Bool);
    // the PyTorch spelling: prometheus.float32, prometheus.int64, ...
    for (const char* n : {"float32", "float64", "float16", "bfloat16", "int64", "int32", "bool"})
        m.attr(n) = m.attr("dtype").attr(n);

    py::enum_<Device>(m, "device")
        .value("cpu",  Device::CPU)
        .value("cuda", Device::CUDA);
    m.attr("cpu") = m.attr("device").attr("cpu");

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init([](std::vector<int64_t> shape, DType dtype) {
                 return std::make_shared<Tensor>(shape, dtype);
             }), py::arg("shape"), py::arg("dtype") = DType::Float32)
        // values arrive as doubles so that int64 ids are exact past 2^24
        .def(py::init([](std::vector<int64_t> shape, std::vector<double> values, DType dtype) {
                 auto t = std::make_shared<Tensor>(shape, dtype);
                 if ((int64_t)values.size() != t->numel())
                     throw std::runtime_error("Tensor: " + std::to_string(values.size()) +
                                              " values for a shape of " +
                                              std::to_string(t->numel()) + " elements");
                 for (int64_t i = 0; i < t->numel(); i++)
                     strided::store(t->storage()->data(), dtype, i, values[i]);
                 return t;
             }), py::arg("shape"), py::arg("values"), py::arg("dtype") = DType::Float32)

        .def_property("data",
                      [](const Tensor& t) { return t.to_vector(); },
                      [](Tensor& t, const std::vector<float>& v) { t.set_data(v); },
                      "Every element in row-major order, as a list (a copy).")
        .def_property_readonly("shape",   [](const Tensor& t) { return t.shape(); })
        .def_property_readonly("strides", [](const Tensor& t) { return t.strides(); })
        .def_property_readonly("dtype",   &Tensor::dtype)
        .def_property_readonly("device",  &Tensor::device)
        .def_property("grad",
                      [](const Tensor& t) { return t.grad_vector(); },
                      [](Tensor& t, const std::vector<float>& g) { t.set_grad(g); })
        .def_property("requires_grad",
                      [](const Tensor& t) { return t.requires_grad(); },
                      [](Tensor& t, bool v) {
                          t.set_requires_grad(v);
                          // the buffer is allocated lazily, so a tensor the
                          // user marks by hand needs one before any op writes
                          if (v) t.ensure_grad();
                      })
        .def("dim",           &Tensor::dim)
        .def("numel",         &Tensor::numel)
        .def("num_el",        &Tensor::num_el)
        .def("size",          &Tensor::size, py::arg("dim"))
        .def("item",          &Tensor::item)
        .def("get",           &Tensor::get)
        .def("set",           &Tensor::set)
        .def("fill",          &Tensor::fill)
        .def("print",         &Tensor::print)
        .def("is_view",       &Tensor::is_view)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("transpose",     &Tensor::transpose, py::arg("dim0") = -2, py::arg("dim1") = -1)
        .def("reshape",       [](Tensor& t, std::vector<int64_t> shape) { return t.reshape(shape); })
        .def("contiguous",    &Tensor::contiguous)
        .def("to",            py::overload_cast<DType>(&Tensor::to))
        .def("to",            py::overload_cast<Device>(&Tensor::to))
        .def("backward",      &Tensor::backward)
        .def("reset_grad",    &Tensor::reset_grad)
        .def("zero_grad",     &Tensor::zero_grad)
        .def("detach",        &Tensor::detach)
        .def("half",          &Tensor::half)
        .def("to_float",      &Tensor::to_float)
        .def_static("zeros", [](std::vector<int64_t> s, DType d) { return Tensor::zeros(s, d); },
                    py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("ones",  [](std::vector<int64_t> s, DType d) { return Tensor::ones(s, d); },
                    py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("full",  [](std::vector<int64_t> s, float v, DType d) { return Tensor::full(s, v, d); },
                    py::arg("shape"), py::arg("value"), py::arg("dtype") = DType::Float32)
        .def_static("randn", [](std::vector<int64_t> s) { return Tensor::randn(s); }, py::arg("shape"))
        .def_static("manual_seed", &Tensor::manual_seed, py::arg("seed"))
        // numpy interop: a copy, in the matching numpy dtype
        .def("numpy", [](TensorPtr t) -> py::object {
            std::vector<py::ssize_t> shape(t->shape().begin(), t->shape().end());
            auto c = t->detach();                       // contiguous copy
            switch (c->dtype()) {
                case DType::Float64: return py::array_t<double>(shape, c->data_ptr<double>());
                case DType::Int64:   return py::array_t<int64_t>(shape, c->data_ptr<int64_t>());
                case DType::Int32:   return py::array_t<int32_t>(shape, c->data_ptr<int32_t>());
                case DType::Bool:    return py::array_t<bool>(shape, c->data_ptr<bool>());
                default: {
                    auto v = c->to_vector();            // float16/bfloat16 widen to float32
                    return py::array_t<float>(shape, v.data());
                }
            }
        })
        .def_static("from_numpy", [](py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                                     std::vector<int64_t> shape) {
            auto buf = arr.request();
            const float* ptr = static_cast<const float*>(buf.ptr);
            return std::make_shared<Tensor>(shape, std::vector<float>(ptr, ptr + buf.size));
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
    m.def("concat_op",     &concat_op, py::arg("inputs"), py::arg("axis") = 0);
    m.def("transpose",     &transpose, py::arg("a"), py::arg("dim0") = -2, py::arg("dim1") = -1,
          "Swap two dimensions: an O(1), differentiable view sharing storage.");
    m.def("permute",       &permute, py::arg("a"), py::arg("dims"),
          "Reorder all dimensions: an O(1), differentiable view.");
    m.def("reshape",       [](TensorPtr a, std::vector<int64_t> shape) { return reshape(a, shape); },
          py::arg("a"), py::arg("shape"),
          "Same elements, new shape (one dim may be -1). A view when the input is contiguous.");
    m.def("contiguous",    &contiguous, py::arg("a"),
          "The tensor in contiguous storage; the input itself if it already is.");
    // Where pip and conda put shared libraries, so `pip install mkl` is found
    // with no configuration: <prefix>/Library/bin on Windows, <prefix>/lib
    // elsewhere, for both the interpreter's prefix and the user base.
    {
        std::vector<std::string> dirs;
        auto sys  = py::module_::import("sys");
        auto site = py::module_::import("site");
        std::vector<std::string> bases = {py::str(sys.attr("prefix"))};
        try { bases.push_back(py::str(site.attr("getuserbase")())); } catch (...) {}
        for (const auto& b : bases) {
#ifdef _WIN32
            dirs.push_back(b + "/Library/bin");
#else
            dirs.push_back(b + "/lib");
#endif
        }
        blas::set_search_dirs(dirs);
    }

    auto backend_info = []() {
        py::dict d;
        d["matmul"]       = matmul_backend_name();
        d["library"]      = blas::library_path();
        d["blas_threads"] = blas::get_threads();
        d["threads"]      = matmul_max_threads();
        return d;
    };
    m.def("backend", backend_info,
          "The matmul backend in use ('openblas', 'mkl' or 'tiled'), the "
          "library file, and thread counts.");
    m.def("set_blas", [backend_info](const std::string& which, const std::string& path) {
        std::string err;
        if (!blas::select(which, path, &err))
            throw std::runtime_error(err);
        return backend_info();
    }, py::arg("which"), py::arg("path") = "",
    "Switch matmul backend: 'auto', 'openblas', 'mkl' or 'tiled'. `path` names "
    "the exact library file if it is not found on its own. Raises if the "
    "library cannot be loaded. Not safe while another thread is training.");
    m.def("linear_op",     &linear_op, py::arg("x"), py::arg("W"),
          py::arg("bias") = nullptr, py::arg("residual") = nullptr,
          "residual + x @ W + bias as one op.");
    m.def("matmul_bt",     &matmul_bt, py::arg("a"), py::arg("b"),
          "a @ b^T without building b^T. a is [m, k], b is [n, k].");

    // ------------------------------------------------------------------------
    // grad mode
    // ------------------------------------------------------------------------
    m.def("is_grad_enabled", []() { return grad_enabled(); },
          "Whether ops currently record a computation graph.");
    m.def("set_grad_enabled", [](bool on) { grad_enabled() = on; },
          py::arg("enabled"),
          "Turn graph recording on or off. Prefer the no_grad() context "
          "manager, which restores the previous setting.");

    py::class_<NoGrad>(m, "_NoGradGuard")
        .def(py::init<>());

    // gradcheck: numeric gradient verification, in the spirit of
    // torch.autograd.gradcheck. Returns a dict so a failing check can report
    // which element disagreed and by how much.
    m.def("gradcheck",
          [](std::function<TensorPtr()> forward, std::vector<TensorPtr> inputs,
             float eps, float atol, float rtol, int max_points, bool verbose) {
              GradCheckOptions o;
              o.eps = eps; o.atol = atol; o.rtol = rtol;
              o.max_points = max_points; o.verbose = verbose;
              GradCheckResult r = gradcheck(forward, inputs, o);
              py::dict d;
              d["passed"]      = r.passed;
              d["checked"]     = r.checked;
              d["failed"]      = r.failed;
              d["max_abs_err"] = r.max_abs_err;
              d["report"]      = r.report;
              return d;
          },
          py::arg("forward"), py::arg("inputs"),
          py::arg("eps") = 1e-2f, py::arg("atol") = 1e-5f, py::arg("rtol") = 2e-2f,
          py::arg("max_points") = 4, py::arg("verbose") = false,
          "Compare analytical gradients against central finite differences. "
          "forward: callable returning a Tensor; inputs: list of Tensors to check. "
          "Returns {passed, checked, failed, max_abs_err, report}.");
    m.def("log_op",        &log_op);
    m.def("exp_op",        &exp_op);
    m.def("pow_op",        &pow_op);
    m.def("sqrt_op",       &sqrt_op);
    m.def("abs_op",        &abs_op);
    m.def("sum",           &sum,     py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false,
          "Sum over dim, or over everything when dim is -1 (unlike PyTorch, where -1 is the last dim).");
    m.def("mean",          &mean,    py::arg("a"), py::arg("dim") = -1, py::arg("keepdim") = false,
          "Mean over dim, or over everything when dim is -1 (unlike PyTorch, where -1 is the last dim).");
    m.def("max_op",        &max_op);
    m.def("min_op",        &min_op);
    m.def("clip",           &clip);
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
        .def_readwrite("training", &Dropout::training)
        .def("eval",  &Dropout::eval)
        .def("train", &Dropout::train);

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

    py::class_<BatchNorm, Module, std::shared_ptr<BatchNorm>>(m, "BatchNorm")
        .def(py::init<int, float, bool, float>(),
             py::arg("num_features"), py::arg("eps") = 1e-5f,
             py::arg("training") = true, py::arg("momentum") = 0.1f)
        .def("forward",    &BatchNorm::forward)
        .def("parameters", &BatchNorm::parameters)
        .def("eval",       &BatchNorm::eval)
        .def("train",      &BatchNorm::train)
        .def_readwrite("training",     &BatchNorm::training)
        .def_readwrite("momentum",     &BatchNorm::momentum)
        .def_readonly("running_mean",  &BatchNorm::running_mean)
        .def_readonly("running_var",   &BatchNorm::running_var);

    py::class_<MultiHeadAttention, Module, std::shared_ptr<MultiHeadAttention>>(m, "MultiHeadAttention")
        .def(py::init<int, int, bool, bool>(),
             py::arg("embed_dim"), py::arg("num_heads"),
             py::arg("causal") = false, py::arg("rope") = false)
        .def("forward",
             static_cast<TensorPtr (MultiHeadAttention::*)(TensorPtr)>(
                 &MultiHeadAttention::forward))
        .def("parameters", &MultiHeadAttention::parameters)
        .def_readonly("W_q", &MultiHeadAttention::W_q)
        .def_readonly("W_k", &MultiHeadAttention::W_k)
        .def_readonly("W_v", &MultiHeadAttention::W_v)
        .def_readonly("W_o", &MultiHeadAttention::W_o);

    py::class_<TransformerBlock, Module, std::shared_ptr<TransformerBlock>>(m, "TransformerBlock")
        .def(py::init<int, int, int, bool, bool>(),
             py::arg("embed_dim"), py::arg("num_heads"), py::arg("ff_dim"),
             py::arg("causal") = false, py::arg("rope") = false)
        .def("forward",
             static_cast<TensorPtr (TransformerBlock::*)(TensorPtr)>(
                 &TransformerBlock::forward))
        .def("parameters", &TransformerBlock::parameters)
        .def_readonly("attn",  &TransformerBlock::attn)
        .def_readonly("norm1", &TransformerBlock::norm1)
        .def_readonly("norm2", &TransformerBlock::norm2)
        .def_readonly("ff1",   &TransformerBlock::ff1)
        .def_readonly("ff2",   &TransformerBlock::ff2);

    py::class_<KVCache>(m, "KVCache")
        .def("reset",    &KVCache::reset)
        .def_readonly("past_len", &KVCache::past_len);

    py::class_<GPT, Module, std::shared_ptr<GPT>>(m, "GPT")
        .def(py::init<int, int, int, int, int, int, bool>(),
             py::arg("vocab_size"), py::arg("max_seq_len"), py::arg("embed_dim"),
             py::arg("num_heads"), py::arg("num_layers"),
             py::arg("ff_dim") = 0, py::arg("rope") = false)
        .def("forward",
             static_cast<TensorPtr (GPT::*)(TensorPtr)>(&GPT::forward))
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
        .def("parameters", &PySequential::parameters)
        .def_readonly("layers", &PySequential::held,
                      "The child Modules in order, so model.layers can be passed to model_summary or the ONNX exporter without keeping a parallel list.");

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
