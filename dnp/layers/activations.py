"""
Activation layers.
"""
from .base import Module
from dnp.core.ops import (
    relu,
    leaky_relu,
    elu,
    softplus,
    swish,
    gelu,
    sigmoid,
    tanh,
    softmax,
)


class _ActivationModule(Module):
    """Base activation function wrapper."""

    def __init__(self, activation_func, name):
        super().__init__()
        self.activation_func = activation_func
        self.__dict__["name"] = name

    def forward(self, x):
        return self.activation_func(x)


class ReLU(_ActivationModule):
    def __init__(self):
        super().__init__(relu, "ReLU")


class Sigmoid(_ActivationModule):
    def __init__(self):
        super().__init__(sigmoid, "Sigmoid")


class Tanh(_ActivationModule):
    def __init__(self):
        super().__init__(tanh, "Tanh")


class Softmax(_ActivationModule):
    def __init__(self):
        super().__init__(softmax, "Softmax")


class LeakyReLU(_ActivationModule):
    def __init__(self, alpha: float = 0.01):
        super().__init__(lambda x: leaky_relu(x, alpha=alpha), f"LeakyReLU(alpha={alpha})")
        self.alpha = alpha


class ELU(_ActivationModule):
    def __init__(self, alpha: float = 1.0):
        super().__init__(lambda x: elu(x, alpha=alpha), f"ELU(alpha={alpha})")
        self.alpha = alpha


class Softplus(_ActivationModule):
    def __init__(self):
        super().__init__(softplus, "Softplus")


class Swish(_ActivationModule):
    def __init__(self):
        super().__init__(swish, "Swish")


class GELU(_ActivationModule):
    def __init__(self):
        super().__init__(gelu, "GELU")
