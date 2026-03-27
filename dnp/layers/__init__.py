"""
Neural Network Layers.
"""

from .base import Module, Sequential
from .activations import (
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU,
    ELU,
    Softplus,
    Swish,
    GELU,
)
from .conv import Conv1d, Conv2d
from .linear import Linear
from .pooling import (
    MaxPool2d,
    AvgPool2d,
    GlobalAvgPool1d,
    GlobalMaxPool1d,
    GlobalAvgPool2d,
    GlobalMaxPool2d,
)
from .normalization import BatchNorm1d, BatchNorm2d, LayerNorm
from .embedding import Embedding
from .regularization import Dropout
from .loss import (
    CrossEntropyLoss,
    MSELoss,
    MAELoss,
    L1Loss,
    HuberLoss,
    SmoothL1Loss,
    LogCoshLoss,
    BCELoss,
    BCEWithLogitsLoss,
    NLLLoss,
    KLDivLoss,
    FocalLoss,
    HingeLoss,
    SquaredHingeLoss,
)
from .utils import Flatten, Reshape, ExpandDims, Squeeze, Repeat, Concatenate, Stack
from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
    PositionalEncoding,
    FeedForward,
    TransformerEncoderLayer,
    FlashAttention,
    RotaryPositionalEncoding,
)
from .advanced import SinkhornTransport, NeuralODE, S4Layer, RNN, LSTM, GRU


__all__ = [
    # Base
    "Module",
    "Sequential",
    # Dense
    "Linear",
    # Conv
    "Conv1d",
    "Conv2d",
    # Pool
    "MaxPool2d",
    "AvgPool2d",
    "GlobalAvgPool1d",
    "GlobalMaxPool1d",
    "GlobalAvgPool2d",
    "GlobalMaxPool2d",
    # Activations
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LeakyReLU",
    "ELU",
    "Softplus",
    "Swish",
    "GELU",
    # Norm
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    # Embedding
    "Embedding",
    # Regularization
    "Dropout",
    # Loss
    "CrossEntropyLoss",
    "MSELoss",
    "MAELoss",
    "L1Loss",
    "HuberLoss",
    "SmoothL1Loss",
    "LogCoshLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "NLLLoss",
    "KLDivLoss",
    "FocalLoss",
    "HingeLoss",
    "SquaredHingeLoss",
    # Utils
    "Flatten",
    "Reshape",
    "ExpandDims",
    "Squeeze",
    "Repeat",
    "Concatenate",
    "Stack",
    # Attention / Transformers
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "SelfAttention",
    "PositionalEncoding",
    "FeedForward",
    "TransformerEncoderLayer",
    "FlashAttention",
    "RotaryPositionalEncoding",
    # Advanced
    "SinkhornTransport",
    "NeuralODE",
    "S4Layer",
    "RNN",
    "LSTM",
    "GRU",
]
