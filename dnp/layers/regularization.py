"""
Regularization layers.
"""
from .base import Module
from dnp.core.tensor import Tensor
from dnp.core import ops


class Dropout(Module):
    """Dropout regularization layer."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        return ops.dropout(x, p=self.p, training=self.training)
