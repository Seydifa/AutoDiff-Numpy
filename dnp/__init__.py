# Local imports: Core modules
from . import ops, core

# Dtype control — primary public API
from .core.backend import set_dtype, get_dtype

# Local imports: Layer classes available at top level
from .layers import (
    Module,
    Sequential,
    Linear,
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    BatchNorm1d,
    BatchNorm2d,
    Dropout,
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
    Flatten,
)

# Local imports: Optimizer classes available at top level
from .core.optimizers import (
    Optimizer,
    SGD,
    Momentum,
    RMSprop,
    Adagrad,
    Adam,
    AdamW,
)

__all__ = [
    "ops",
    "core",
    # Dtype control
    "set_dtype",
    "get_dtype",
    # Base classes
    "Module",
    "Sequential",
    # Linear layers
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
