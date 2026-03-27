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
        name: str | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        if name is not None:
            self.__dict__["_instance_name"] = name

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
        """Forward pass: (..., in_features) -> (..., out_features)."""
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"{self.__class__.__name__}: expected last dimension {self.in_features}, "
                f"got {x.shape[-1]} (input shape {x.shape})"
            )
        y = ops.matmul(x, self.W)
        if self.use_bias:
            y = y + self.b
        return y

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.use_bias})"
        )
