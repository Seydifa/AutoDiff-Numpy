"""
Utility modules.
"""

from .base import Module
from dnp.core.tensor import Tensor
from dnp.core import ops


class Flatten(Module):
    """Flattens a tensor along specified dimensions."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        ndim = len(shape)
        start = self.start_dim % ndim
        end = self.end_dim % ndim

        if start > end:
            raise ValueError(
                f"Invalid range [{start}, {end}] for flattening tensor of ndim {ndim}."
            )

        new_shape = shape[:start] + (-1,) + shape[end + 1 :]
        return ops.reshape(x, newshape=new_shape)


class Reshape(Module):
    """Reshapes a tensor to a target shape."""

    def __init__(self, *shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x: Tensor) -> Tensor:
        return ops.reshape(x, newshape=self.shape)


class ExpandDims(Module):
    """Inserts a size-1 dimension at the given axis."""

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return ops.expand_dims(x, axis=self.axis)


class Squeeze(Module):
    """Removes size-1 dimensions from a tensor."""

    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return ops.squeeze(x, axis=self.axis)


class Repeat(Module):
    """Repeats tensor elements along an axis."""

    def __init__(self, repeats, axis=None):
        super().__init__()
        self.repeats = repeats
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return ops.repeat(x, repeats=self.repeats, axis=self.axis)


class Concatenate(Module):
    """Concatenates tensors along an axis."""

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def forward(self, *xs: Tensor) -> Tensor:
        return ops.concatenate(*xs, axis=self.axis)


class Stack(Module):
    """Stacks tensors along a new axis."""

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def forward(self, *xs: Tensor) -> Tensor:
        return ops.stack(*xs, axis=self.axis)
