"""
Normalization layers.
"""

from .base import Module
from dnp.core.tensor import Tensor
from dnp.core import ops
from dnp.core.session import session


class BatchNorm1d(Module):
    """Batch normalization for 1D (per-feature) normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.weight = self.add_weight(
            "weight", shape=(num_features,), initializer="ones"
        )
        self.bias = self.add_weight("bias", shape=(num_features,), initializer="zeros")

        self.running_mean = self.add_weight(
            "running_mean",
            shape=(num_features,),
            initializer="zeros",
            trainable=False,
        )
        self.running_var = self.add_weight(
            "running_var",
            shape=(num_features,),
            initializer="ones",
            trainable=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = ops.mean(x, axis=0)
            diff = x - mean
            var = ops.mean(ops.square(diff), axis=0)
            mean_for_norm = mean.detach()
            var_for_norm = var.detach()
            with session.no_grad():
                self.running_mean = (
                    1.0 - self.momentum
                ) * self.running_mean + self.momentum * mean_for_norm
                self.running_var = (
                    1.0 - self.momentum
                ) * self.running_var + self.momentum * var_for_norm
        else:
            mean_for_norm = self.running_mean
            var_for_norm = self.running_var

        return ops.batch_norm(
            x,
            self.weight,
            self.bias,
            mean=mean_for_norm,
            var=var_for_norm,
            eps=self.eps,
        )


class BatchNorm2d(Module):
    """Batch normalization for 2D (per-channel) normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.weight = self.add_weight(
            "weight", shape=(num_features,), initializer="ones"
        )
        self.bias = self.add_weight("bias", shape=(num_features,), initializer="zeros")

        self.running_mean = self.add_weight(
            "running_mean",
            shape=(num_features,),
            initializer="zeros",
            trainable=False,
        )
        self.running_var = self.add_weight(
            "running_var",
            shape=(num_features,),
            initializer="ones",
            trainable=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = ops.mean(x, axis=(0, 2, 3))
            mean_r = ops.reshape(mean, newshape=(1, self.num_features, 1, 1))
            diff = x - mean_r
            var = ops.mean(ops.square(diff), axis=(0, 2, 3))
            mean_for_norm = mean.detach()
            var_for_norm = var.detach()
            with session.no_grad():
                self.running_mean = (
                    1.0 - self.momentum
                ) * self.running_mean + self.momentum * mean_for_norm
                self.running_var = (
                    1.0 - self.momentum
                ) * self.running_var + self.momentum * var_for_norm
        else:
            mean_for_norm = self.running_mean
            var_for_norm = self.running_var

        shape_for_norm = (1, self.num_features, 1, 1)
        mean_r = ops.reshape(mean_for_norm, newshape=shape_for_norm)
        var_r = ops.reshape(var_for_norm, newshape=shape_for_norm)

        weight_r = ops.reshape(self.weight, newshape=shape_for_norm)
        bias_r = ops.reshape(self.bias, newshape=shape_for_norm)

        return ops.batch_norm(x, weight_r, bias_r, mean=mean_r, var=var_r, eps=self.eps)


class LayerNorm(Module):
    """Layer normalization."""

    def __init__(self, ndim, bias=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = self.add_weight("gamma", shape=(ndim,), initializer="ones")
        self.use_bias = bias
        if bias:
            self.beta = self.add_weight("beta", shape=(ndim,), initializer="zeros")

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.mean(x, axis=-1, keepdims=True)
        diff = x - mean
        var = ops.mean(ops.square(diff), axis=-1, keepdims=True)
        std = ops.sqrt(var + float(self.eps))
        x_norm = diff / std
        out = x_norm * self.gamma
        if self.use_bias:
            out = out + self.beta
        return out
