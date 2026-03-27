# dnp/autograd/session.py
# Backward-compatibility re-export — logic now lives in dnp/core.
import warnings

warnings.warn(
    "dnp.autograd.session is deprecated. Use dnp.core.session instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Local imports
from dnp.core.session import SessionGraph, session
from dnp.core.tensor import Tensor
from dnp.core.vjp_rules import VJP_RULES

__all__ = ["SessionGraph", "session", "Tensor", "VJP_RULES"]
