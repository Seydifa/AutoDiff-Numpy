# Local imports: Backward-compatibility re-export from core/layers.py
# All layer implementations have been consolidated in dnp/core/layers.py
from dnp.core.layers import (
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

__all__ = [
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
]
