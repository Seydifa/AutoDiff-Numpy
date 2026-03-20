# Local imports: Core modules
from . import ops, core

# Dtype control — primary public API
from .core.backend import set_dtype, get_dtype

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
