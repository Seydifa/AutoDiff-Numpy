"""
Linear / Dense layers.
"""

from .base import Module
from dnp.core.tensor import Tensor
from dnp.core import ops
from dnp.core.backend import backend


class Linear(Module):
    """Fully-connected (dense) layer: y = Wx + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str = "Linear",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.__dict__["name"] = name

        std = float(backend.sqrt(2.0 / (in_features + out_features)))

        self.W = self.add_weight(
            "W",
            shape=(in_features, out_features),
            initializer=lambda s: backend.random.randn(*s) * std,
        )
        if bias:
            self.b = self.add_weight("b", shape=(out_features,), initializer="zeros")
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: (batch, in_features) -> (batch, out_features)."""
        y = ops.matmul(x, self.W)
        if self.use_bias:
            y = y + self.b
        return y
