# Local imports: Core modules
from . import ops, core

# Dtype and device control — primary public API
from .core.backend import set_dtype, get_dtype, set_device

# Tensor and session — required for any computation
from .core.tensor import Tensor
from .core.session import session, graph_fn

# Local imports: Layer classes available at top level
from .layers import (
    Module,
    Sequential,
    Linear,
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    GlobalAvgPool1d,
    GlobalMaxPool1d,
    GlobalAvgPool2d,
    GlobalMaxPool2d,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU,
    ELU,
    Softplus,
    Swish,
    GELU,
    BatchNorm1d,
    BatchNorm2d,
    LayerNorm,
    Dropout,
    Embedding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
    PositionalEncoding,
    FeedForward,
    TransformerEncoderLayer,
    Flatten,
    Reshape,
    ExpandDims,
    Squeeze,
    Repeat,
    Concatenate,
    Stack,
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

# Local imports: Optimizer classes available at top level
from .core.optimizers import (
    Optimizer,
    SGD,
    Momentum,
    RMSprop,
    Adagrad,
    Adam,
    AdamW,
    # Schedulers (v3)
    LRScheduler,
    StepLR,
    MultiStepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    WarmupScheduler,
)

# Training utilities (v3)
from .utils import (
    Callback,
    EarlyStopping,
    History,
    ModelCheckpoint,
    ProgressLogger,
    Trainer,
)

__all__ = [
    "ops",
    "core",
    # Dtype control
    "set_dtype",
    "get_dtype",
    # Core autograd
    "Tensor",
    "session",
    "graph_fn",
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
    "GlobalAvgPool1d",
    "GlobalMaxPool1d",
    "GlobalAvgPool2d",
    "GlobalMaxPool2d",
    # Activation layers
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LeakyReLU",
    "ELU",
    "Softplus",
    "Swish",
    "GELU",
    # Normalization layers
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    # Regularization layers
    "Dropout",
    # Embedding layers
    "Embedding",
    # Attention layers
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "SelfAttention",
    # Transformer building blocks
    "PositionalEncoding",
    "FeedForward",
    "TransformerEncoderLayer",
    # Utility layers
    "Flatten",
    "Reshape",
    "ExpandDims",
    "Squeeze",
    "Repeat",
    "Concatenate",
    "Stack",
    # Loss functions
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
    # Optimization algorithms
    "Optimizer",
    "SGD",
    "Momentum",
    "RMSprop",
    "Adagrad",
    "Adam",
    "AdamW",
    # LR Schedulers
    "LRScheduler",
    "StepLR",
    "MultiStepLR",
    "CosineAnnealingLR",
    "ReduceLROnPlateau",
    "WarmupScheduler",
    # Training utilities
    "Callback",
    "EarlyStopping",
    "History",
    "ModelCheckpoint",
    "ProgressLogger",
    "Trainer",
]
