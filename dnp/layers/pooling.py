"""
Pooling layers.
"""

from .base import Module
from dnp.core.tensor import Tensor
from dnp.core.ops import avg_pool2d, max as reduce_max, max_pool2d, mean as reduce_mean


class MaxPool2d(Module):
    """2D max pooling."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        self.stride = (
            self.kernel_size
            if stride is None
            else ((stride, stride) if isinstance(stride, int) else tuple(stride))
        )
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return max_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )


class AvgPool2d(Module):
    """2D average pooling."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        self.stride = (
            self.kernel_size
            if stride is None
            else ((stride, stride) if isinstance(stride, int) else tuple(stride))
        )
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return avg_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )


class GlobalAvgPool1d(Module):
    """Global average pooling over the last spatial dimension."""

    def __init__(self, keepdims=False):
        super().__init__()
        self.keepdims = keepdims

    def forward(self, x: Tensor) -> Tensor:
        return reduce_mean(x, axis=2, keepdims=self.keepdims)


class GlobalMaxPool1d(Module):
    """Global max pooling over the last spatial dimension."""

    def __init__(self, keepdims=False):
        super().__init__()
        self.keepdims = keepdims

    def forward(self, x: Tensor) -> Tensor:
        return reduce_max(x, axis=2, keepdims=self.keepdims)


class GlobalAvgPool2d(Module):
    """Global average pooling over the spatial height/width dimensions."""

    def __init__(self, keepdims=False):
        super().__init__()
        self.keepdims = keepdims

    def forward(self, x: Tensor) -> Tensor:
        return reduce_mean(x, axis=(2, 3), keepdims=self.keepdims)


class GlobalMaxPool2d(Module):
    """Global max pooling over the spatial height/width dimensions."""

    def __init__(self, keepdims=False):
        super().__init__()
        self.keepdims = keepdims

    def forward(self, x: Tensor) -> Tensor:
        return reduce_max(x, axis=(2, 3), keepdims=self.keepdims)
