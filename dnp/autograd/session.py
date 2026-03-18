# dnp/autograd/session.py
# Backward-compatibility re-export — logic now lives in dnp/core.

# Local imports
from dnp.core.session import SessionGraph, session
from dnp.core.tensor import Tensor
from dnp.core.vjp_rules import VJP_RULES

__all__ = ["SessionGraph", "session", "Tensor", "VJP_RULES"]
