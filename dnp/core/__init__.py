"""
dnp/core/__init__.py
====================
Public API of the consolidated core module.

import dnp.core as core
core.Tensor(...)
core.add(x, y)
core.session.reset()
core.Linear(784, 128)
"""

# Local imports: Core autograd engine
from .vjp_rules import VJP_RULES, EPSILON, unbroadcast
from .session import SessionGraph, session
from .tensor import Tensor
from .ops import (
    Ops,
    add,
    subtract,
    multiply,
    divide,
    power,
    maximum,
    minimum,
    matmul,
    dot,
    negative,
    square,
    sqrt,
    exp,
    log,
    log1p,
    expm1,
    absolute,
    sign,
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    floor,
    ceil,
    round,
    sum,
    mean,
    prod,
    max,
    min,
    reshape,
    transpose,
    expand_dims,
    squeeze,
    conv2d,
    sigmoid,
    relu,
    leaky_relu,
    elu,
    softplus,
    swish,
    gelu,
    softmax,
    max_pool2d,
    avg_pool2d,
    dropout,
    batch_norm,
)

# Local imports: Neural network layers (centralized in core/layers.py)
from .layers import (
    Module,
    Linear,
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    Flatten,
    Sequential,
    BatchNorm1d,
    BatchNorm2d,
    Dropout,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
)

# Local imports: Optimization algorithms
from .optimizers import (
    Optimizer,
    SGD,
    Momentum,
    RMSprop,
    Adagrad,
    Adam,
    AdamW,
)

__all__ = [
    # Autograd engine
    "SessionGraph",
    "session",
    "Tensor",
    # VJP infrastructure
    "VJP_RULES",
    "EPSILON",
    "unbroadcast",
    # Ops class
    "Ops",
    # All operation instances
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "maximum",
    "minimum",
    "matmul",
    "dot",
    "negative",
    "square",
    "sqrt",
    "exp",
    "log",
    "log1p",
    "expm1",
    "absolute",
    "sign",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "floor",
    "ceil",
    "round",
    "sum",
    "mean",
    "prod",
    "max",
    "min",
    "reshape",
    "transpose",
    "expand_dims",
    "squeeze",
    "conv2d",
    "sigmoid",
    "relu",
    "leaky_relu",
    "elu",
    "softplus",
    "swish",
    "gelu",
    "softmax",
    # Pooling operations (vectorized)
    "max_pool2d",
    "avg_pool2d",
    # Regularization operations
    "dropout",
    # Normalization operations
    "batch_norm",
    # Neural-network layers
    "Module",
    "Sequential",
    # Linear/Dense layers
    "Linear",
    # Convolutional layers
    "Conv2d",
    # Pooling layers
    "MaxPool2d",
    "AvgPool2d",
    # Activation layers
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    # Normalization layers
    "BatchNorm1d",
    "BatchNorm2d",
    # Regularization layers
    "Dropout",
    # Attention layers
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "SelfAttention",
    # Utility layers
    "Flatten",
    # Optimization algorithms
    "Optimizer",
    "SGD",
    "Momentum",
    "RMSprop",
    "Adagrad",
    "Adam",
    "AdamW",
]
