# dnp/autograd/__init__.py
# Backward-compatibility re-export — logic now lives in dnp/core.
import warnings

warnings.warn(
    "dnp.autograd is deprecated. Use dnp.core (or top-level dnp) instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Local imports: Core autograd components
from dnp.core.session import SessionGraph, session
from dnp.core.tensor import Tensor

__all__ = ["SessionGraph", "session", "Tensor"]
