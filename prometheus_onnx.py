"""
ONNX export utility for prometheus models.

ONNX (Open Neural Network Exchange) is a standard file format for ML models.
Once exported, the model can be loaded and run by other frameworks such as
ONNX Runtime, TensorFlow, or deployed to edge devices — without needing
prometheus or Python at all.

Usage:
    from prometheus_onnx import save_onnx

    layers = [p.Linear(3, 8), p.ReLU(), p.Linear(8, 1), p.Sigmoid()]
    model  = p.Sequential(layers)
    save_onnx(layers, input_shape=[1, 3], path="model.onnx")

Currently supported layers:
    Linear   → ONNX Gemm  (matrix multiply + bias)
    ReLU     → ONNX Relu
    Sigmoid  → ONNX Sigmoid
    Tanh     → ONNX Tanh
    Softmax  → ONNX Softmax
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import prometheus as p


# Maps each prometheus activation layer name to its equivalent ONNX operator.
# Activation layers have no weights — they just apply a function element-wise.
_ACTIVATION_OPS = {
    "ReLU":    "Relu",
    "Sigmoid": "Sigmoid",
    "Tanh":    "Tanh",
    "Softmax": "Softmax",
}


def save_onnx(layers, input_shape, path="model.onnx"):
    """
    Export a prometheus model to an ONNX file.

    An ONNX model is a computation graph: a list of nodes (operations) connected
    by named tensors. This function walks your layers one by one and converts each
    into the equivalent ONNX node, collecting the trained weights as constants.

    Args:
        layers (list):       The same list of layer objects you passed to Sequential.
                             e.g. [p.Linear(2, 32), p.ReLU(), p.Linear(32, 1)]
        input_shape (list):  Shape of a single input batch.
                             e.g. [1, 2] means batch_size=1, features=2.
        path (str):          File path to write the .onnx file to.
                             Defaults to "model.onnx".

    Raises:
        ValueError: If a layer type is not yet supported for ONNX export.

    Example:
        layers = [p.Linear(2, 16), p.ReLU(), p.Linear(16, 1), p.Sigmoid()]
        save_onnx(layers, input_shape=[1, 2], path="my_model.onnx")
    """

    nodes = []         # the computation steps (ops) in order
    initializers = []  # constant tensors (weights, biases) stored in the file
    node_idx = 0       # counter used to give each node a unique name
    current = "input"  # name of the tensor flowing into the next layer

    for layer in layers:
        name = type(layer).__name__          # e.g. "Linear", "ReLU"
        node_name = f"{name}_{node_idx}"    # unique name for this node, e.g. "Linear_0"
        output = f"output_{node_idx}"       # name for this node's output tensor

        if name == "Linear":
            # A Linear layer computes: output = input @ W + b
            # We read the trained weights and bias out of the layer and store
            # them as ONNX initializers (constant tensors embedded in the file).
            W = np.array(layer.weights.data, dtype=np.float32).reshape(
                layer.weights.shape[0], layer.weights.shape[1]
            )
            b = np.array(layer.bias.data, dtype=np.float32)

            w_name = f"{node_name}_W"  # unique name for the weight tensor
            b_name = f"{node_name}_b"  # unique name for the bias tensor

            # Register the weight and bias as constants inside the ONNX graph
            initializers.append(numpy_helper.from_array(W, name=w_name))
            initializers.append(numpy_helper.from_array(b, name=b_name))

            # Gemm = General Matrix Multiply. ONNX's standard op for linear layers.
            # transB=0 means W is already in the right orientation (no transpose needed).
            nodes.append(helper.make_node(
                "Gemm",
                inputs=[current, w_name, b_name],
                outputs=[output],
                name=node_name,
                transB=0,
            ))

        elif name in _ACTIVATION_OPS:
            # Activation layers have no weights — they just transform their input.
            # Softmax needs axis=1 so it normalises across features, not across the batch.
            kwargs = {"axis": 1} if name == "Softmax" else {}
            nodes.append(helper.make_node(
                _ACTIVATION_OPS[name],
                inputs=[current],
                outputs=[output],
                name=node_name,
                **kwargs,
            ))

        else:
            raise ValueError(
                f"save_onnx: unsupported layer type '{name}'. "
                f"Supported: Linear, ReLU, Sigmoid, Tanh, Softmax."
            )

        # The output of this layer becomes the input to the next one
        current = output
        node_idx += 1

    # ── Determine the final output shape ────────────────────────────────────
    # ONNX requires us to declare the shape of the graph's output tensor.
    # We replay the shape through each Linear layer (activations don't change shape).
    out_shape = list(input_shape)
    for layer in layers:
        if type(layer).__name__ == "Linear":
            # Linear maps (batch, in_features) → (batch, out_features)
            out_shape = [out_shape[0], layer.weights.shape[1]]

    # ── Build the ONNX graph ─────────────────────────────────────────────────
    # Declare the graph's input and output tensors with their shapes and dtype.
    input_info  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, input_shape)
    output_info = helper.make_tensor_value_info(current,  TensorProto.FLOAT, out_shape)

    # A graph bundles: the ops (nodes), input/output declarations, and constants (initializers)
    graph = helper.make_graph(
        nodes,
        "prometheus_model",
        [input_info],
        [output_info],
        initializer=initializers,
    )

    # Wrap the graph in a top-level ONNX model using opset 17 (a recent stable version)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.doc_string = "Exported from prometheus ML library"

    # Validate the graph structure before saving — catches shape mismatches etc.
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"saved to {path}  ({len(layers)} layers, opset 17)")
