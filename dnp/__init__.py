# Local imports: Core modules
import warnings as _warnings
from . import core


# Lazy import of the deprecated shim subpackage.
# Using __getattr__ means it only loads (and warns) when accessed directly.
def __getattr__(name):
    if name == "ops":
        import importlib as _il

        return _il.import_module("dnp.ops")
    raise AttributeError(f"module 'dnp' has no attribute {name!r}")


# Dtype and device control — primary public API
from .core.backend import set_dtype, get_dtype, set_device

# Tensor and session — required for any computation
from .core.tensor import Tensor
from .core.session import session, graph_fn

# Layer classes — wildcard so new layers only need dnp/layers/__init__.py updates.
from .layers import *  # noqa: F401, F403
from .layers import __all__ as _layers_all

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
    DataLoader,
    clip_grad_norm_,
    clip_grad_value_,
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
    # All layer classes (auto-synced with dnp/layers/__all__)
    *_layers_all,
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
    "DataLoader",
    "clip_grad_norm_",
    "clip_grad_value_",
]
