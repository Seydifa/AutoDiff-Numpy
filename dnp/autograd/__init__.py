# dnp/autograd/__init__.py
# Backward-compatibility re-export — logic now lives in dnp/core.

# Local imports: Core autograd components
from dnp.core.session import SessionGraph, session
from dnp.core.tensor import Tensor

__all__ = ["SessionGraph", "session", "Tensor"]
