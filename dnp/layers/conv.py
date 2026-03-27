"""
Convolutional layers.
"""

from .base import Module
from dnp.core.tensor import Tensor
from dnp.core import ops
from dnp.core.backend import backend


class Conv2d(Module):
    """2D convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = padding
        self.use_bias = bias

        kH, kW = self.kernel_size
        std = float(backend.sqrt(2.0 / (in_channels * kH * kW)))

        self.W = self.add_weight(
            "W",
            shape=(out_channels, in_channels, kH, kW),
            initializer=lambda s: backend.random.randn(*s) * std,
        )
        if bias:
            self.b = self.add_weight("b", shape=(out_channels,), initializer="zeros")
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using vectorized im2col + matmul kernel."""
        batch_size, in_channels, H, W = x.shape
        out_channels, _, kH, kW = self.W.shape
        stride_h, stride_w = self.stride
        pad = self.padding

        if isinstance(pad, str):
            if pad.lower() == "same":
                pad_h, pad_w = (kH - 1) // 2, (kW - 1) // 2
            elif pad.lower() == "valid":
                pad_h = pad_w = 0
            else:
                raise ValueError(f"Unknown padding mode: {pad}")
        elif isinstance(pad, (tuple, list)):
            pad_h, pad_w = int(pad[0]), int(pad[1])
        else:
            pad_h = pad_w = int(pad)

        y = ops.conv2d_nd(
            x, self.W, stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w
        )

        if self.use_bias:
            bias_reshaped = ops.reshape(self.b, newshape=(1, self.out_channels, 1, 1))
            y = y + bias_reshaped
        return y


class Conv1d(Module):
    """1D convolution layer implemented via 2D projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding=0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        std = float(backend.sqrt(2.0 / (in_channels * kernel_size)))

        self.W = self.add_weight(
            "W",
            shape=(out_channels, in_channels, kernel_size),
            initializer=lambda s: backend.random.randn(*s) * std,
        )
        if bias:
            self.b = self.add_weight("b", shape=(out_channels,), initializer="zeros")
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using 2D dimensional projection."""
        # x is (batch, in_channels, L) -> expand to (batch, in_channels, 1, L)
        x_2d = ops.expand_dims(x, axis=2)

        # W is (out_channels, in_channels, kL) -> expand to (out_channels, in_channels, 1, kL)
        w_2d = ops.expand_dims(self.W, axis=2)

        if isinstance(self.padding, str):
            if self.padding.lower() == "same":
                pad_w = (self.kernel_size - 1) // 2
            elif self.padding.lower() == "valid":
                pad_w = 0
            else:
                raise ValueError(f"Unknown padding mode: {self.padding}")
        else:
            pad_w = int(self.padding)

        y_2d = ops.conv2d_nd(
            x_2d, w_2d, stride_h=1, stride_w=self.stride, pad_h=0, pad_w=pad_w
        )

        # y_2d is (batch, out_channels, 1, L_out) -> squeeze back to 1D
        y = ops.squeeze(y_2d, axis=2)

        if self.use_bias:
            bias_reshaped = ops.reshape(self.b, newshape=(1, self.out_channels, 1))
            y = y + bias_reshaped
        return y
