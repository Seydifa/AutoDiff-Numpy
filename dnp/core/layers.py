"""
dnp/core/layers.py
==================
Central hub for all neural network layer implementations.

This module consolidates imports from dnp/layers/ subdirectory,
providing a single, organized access point to all layer types.

Organization:
  - Base classes
  - Linear/Dense layers
  - Convolutional layers
  - Pooling layers
  - Activation layers
  - Normalization layers
  - Regularization layers
  - Attention layers
  - Utilities
"""

# Third-party libraries
import numpy as np
from .backend import get_xp, as_numpy, get_dtype

# Local imports: Core tensor and operations
from .tensor import Tensor
from . import ops
from .vjp_rules import _conv2d_forward_kernel
from .ops import (
    matmul,
    add,
    relu,
    sigmoid,
    tanh,
    softmax,
    max_pool2d,
    avg_pool2d,
    dropout,
    batch_norm,
)

# ============================================================================
# BASE CLASSES
# ============================================================================


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        # Use __dict__ directly to avoid triggering __setattr__ recursively.
        self.__dict__["_parameters"] = {}
        self.__dict__["_modules"] = {}

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix=""):
        for name, param in self._parameters.items():
            full = f"{prefix}.{name}" if prefix else name
            yield full, param
        for mod_name, module in self._modules.items():
            sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            yield from module.named_parameters(sub_prefix)

    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0.0)  # in-place; stays on the same device (CPU or GPU)

    def cpu(self):
        """Move all parameters in this module to the CPU."""
        for name, param in self._parameters.items():
            param.cpu()
        for name, module in self._modules.items():
            module.cpu()
        return self

    def cuda(self):
        """Move all parameters in this module to the GPU (CuPy)."""
        for name, param in self._parameters.items():
            param.cuda()
        for name, module in self._modules.items():
            module.cuda()
        return self

    def eval(self):
        """Set module to evaluation mode (disables dropout, uses running stats for batchnorm)."""
        self.training = False
        for module in self._modules.values():
            module.eval()
        return self

    def train(self):
        """Set module to training mode (enables dropout, updates running stats for batchnorm)."""
        self.training = True
        for module in self._modules.values():
            module.train()
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "The forward() method must be implemented in subclasses."
        )

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for name, module in self._modules.items():
            module_repr = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {module_repr}")
        lines.append(")")
        return "\n".join(lines)


class Sequential(Module):
    """Container that executes modules in sequence."""

    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class _ActivationModule(Module):
    """Base activation function wrapper."""

    def __init__(self, activation_func, name):
        super().__init__()
        self.activation_func = activation_func
        self.__dict__["name"] = name

    def forward(self, x):
        return self.activation_func(x)


# ============================================================================
# LINEAR / DENSE LAYERS
# ============================================================================


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

        std = np.sqrt(2.0 / (in_features + out_features))

        self.W = Tensor(
            np.random.randn(in_features, out_features) * std,
            name=f"{self.name}_Poids",
        )
        if bias:
            self.b = Tensor(np.zeros(out_features), name=f"{self.name}_Biais")
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: (batch, in_features) -> (batch, out_features)."""
        y = x @ self.W
        if self.use_bias:
            y = y + self.b
        return y


# ============================================================================
# CONVOLUTIONAL LAYERS
# ============================================================================


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
        std = np.sqrt(2.0 / (in_channels * kH * kW))

        self.W = Tensor(
            np.random.randn(out_channels, in_channels, kH, kW) * std,
            name="Conv2d_poids",
        )
        if bias:
            self.b = Tensor(np.zeros(out_channels), name="Conv2d_biais")
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using vectorized im2col + matmul kernel."""
        batch_size, in_channels, H, W = x.shape
        out_channels, _, kH, kW = self.W.shape
        stride_h, stride_w = self.stride
        pad = self.padding

        # Handle string padding modes
        if isinstance(pad, str):
            if pad.lower() == "same":
                pad_h, pad_w = (kH - 1) // 2, (kW - 1) // 2
            elif pad.lower() == "valid":
                pad_h = pad_w = 0
            else:
                raise ValueError(f"Unknown padding mode: {pad}")
        else:
            pad_h = pad_w = pad

        # Apply padding — stays on the input device (CPU or GPU)
        xp = get_xp(x.data)
        if pad_h > 0 or pad_w > 0:
            x_padded = xp.pad(
                x.data,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            )
        else:
            x_padded = x.data

        # Compute output spatial dimensions
        H_out = (x_padded.shape[2] - kH) // stride_h + 1
        W_out = (x_padded.shape[3] - kW) // stride_w + 1

        y_data = _conv2d_forward_kernel(
            x_padded, self.W.data, stride_h, stride_w, H_out, W_out
        )

        # Wrap result in Tensor.
        # op_func = _conv2d_forward_kernel so the VJP rule can propagate
        # gradients back to x (input) and self.W (filter weights).
        # op_kwargs stores everything the backward pass needs to reconstruct
        # x_padded and compute dx/dW via the col2im scatter-add.
        y = Tensor(
            y_data,
            parents=[x, self.W],
            op_func=_conv2d_forward_kernel,
            op_kwargs={
                "pad_h": pad_h,
                "pad_w": pad_w,
                "stride_h": stride_h,
                "stride_w": stride_w,
                "H_out": H_out,
                "W_out": W_out,
            },
            name="Conv2d_forward",
        )

        if self.use_bias:
            bias_reshaped = self.b.reshape(1, out_channels, 1, 1)
            y = y + bias_reshaped
        return y


# ============================================================================
# POOLING LAYERS
# ============================================================================


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
        """Forward pass using vectorized max pooling."""
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
        """Forward pass using vectorized average pooling."""
        return avg_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )


# ============================================================================
# ACTIVATION LAYERS
# ============================================================================


class ReLU(_ActivationModule):
    """Rectified Linear Unit activation."""

    def __init__(self):
        super().__init__(relu, "ReLU")


class Sigmoid(_ActivationModule):
    """Sigmoid activation."""

    def __init__(self):
        super().__init__(sigmoid, "Sigmoid")


class Tanh(_ActivationModule):
    """Hyperbolic tangent activation."""

    def __init__(self):
        super().__init__(tanh, "Tanh")


class Softmax(_ActivationModule):
    """Softmax activation."""

    def __init__(self):
        super().__init__(softmax, "Softmax")


# ============================================================================
# NORMALIZATION LAYERS
# ============================================================================


class BatchNorm1d(Module):
    """Batch normalization for 1D (per-feature) normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.weight = Tensor(np.ones(num_features), name="BN1d.weight")  # γ
        self.bias = Tensor(np.zeros(num_features), name="BN1d.bias")  # β

        self.running_mean = np.zeros(num_features, dtype=get_dtype())
        self.running_var = np.ones(num_features, dtype=get_dtype())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using batch norm operation."""
        xp = get_xp(x.data)
        if self.training:
            mean = xp.mean(x.data, axis=0)
            var = xp.var(x.data, axis=0)
            # Running stats are always kept as plain numpy (not differentiable).
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * as_numpy(mean)
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * as_numpy(var)
        else:
            mean = xp.asarray(self.running_mean)
            var = xp.asarray(self.running_var)

        # Pass mean/var/eps as kwargs so they land in op_kwargs and are
        # available to the VJP during the backward pass.
        return batch_norm(x, self.weight, self.bias, mean=mean, var=var, eps=self.eps)


class BatchNorm2d(Module):
    """Batch normalization for 2D (per-channel) normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.weight = Tensor(np.ones(num_features), name="BN2d.weight")  # γ
        self.bias = Tensor(np.zeros(num_features), name="BN2d.bias")  # β

        self.running_mean = np.zeros(num_features, dtype=get_dtype())
        self.running_var = np.ones(num_features, dtype=get_dtype())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using batch norm operation."""
        xp = get_xp(x.data)
        if self.training:
            mean = xp.mean(x.data, axis=(0, 2, 3))
            var = xp.var(x.data, axis=(0, 2, 3))
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * as_numpy(mean)
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * as_numpy(var)
        else:
            mean = xp.asarray(self.running_mean)
            var = xp.asarray(self.running_var)

        # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
        shape_for_norm = (1, self.num_features, 1, 1)
        mean_r = mean.reshape(shape_for_norm)
        var_r = var.reshape(shape_for_norm)

        # ops.reshape keeps weight and bias connected to the graph so their
        # gradients are properly accumulated during backward.
        weight_r = ops.reshape(self.weight, newshape=shape_for_norm)
        bias_r = ops.reshape(self.bias, newshape=shape_for_norm)

        # Pass mean/var/eps as kwargs so they're stored in op_kwargs for VJP.
        return batch_norm(x, weight_r, bias_r, mean=mean_r, var=var_r, eps=self.eps)


class LayerNorm(Module):
    """Layer normalization."""

    def __init__(self, ndim, bias=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = Tensor(np.ones(ndim), name="LN_gamma")
        self.use_bias = bias
        if bias:
            self.beta = Tensor(np.zeros(ndim), name="LN_beta")

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.mean(x, axis=-1, keepdims=True)
        diff = x - mean
        # ops.square (not diff**2) — the scalar exponent 2 is not a Tensor and
        # therefore not stored as a parent, which breaks the np.power VJP.
        var = ops.mean(ops.square(diff), axis=-1, keepdims=True)
        eps_t = Tensor(np.array([self.eps]))  # float64 — matches default tensor dtype
        std = ops.sqrt(var + eps_t)
        x_norm = diff / std
        out = x_norm * self.gamma
        if self.use_bias:
            out = out + self.beta
        return out


class Embedding(Module):
    """Embedding layer constructed with one-hot encoding for autograd compatibility."""

    def __init__(self, num_embeddings, embedding_dim, name="Embedding"):
        super().__init__()
        self.lin = Linear(num_embeddings, embedding_dim, bias=False, name=name)
        self.num_embeddings = num_embeddings

    def forward(self, x_idx):
        x_np = as_numpy(x_idx.data if isinstance(x_idx, Tensor) else x_idx).astype(int)
        one_hot = np.eye(self.num_embeddings, dtype=np.float32)[x_np]
        # Inherit device from embedding weight so GPU models stay on GPU
        return self.lin(Tensor(one_hot, device=self.lin.W.device))


# ============================================================================
# REGULARIZATION LAYERS
# ============================================================================


class Dropout(Module):
    """Dropout regularization layer."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using dropout operation."""
        return dropout(x, p=self.p, training=self.training)


# ============================================================================
# ATTENTION LAYERS
# ============================================================================


class ScaledDotProductAttention(Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.scale = 1.0 / np.sqrt(d_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None) -> Tensor:
        """
        Args:
            query: (..., seq_len_q, d_k)
            key: (..., seq_len_k, d_k)
            value: (..., seq_len_v, d_v)
            mask: (..., seq_len_q, seq_len_k) or None
        Returns:
            output: (..., seq_len_q, d_v)
        """
        # Transpose last two dims of key — stays in the graph via ops.transpose.
        ndim = key.data.ndim
        axes = list(range(ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        key_T = ops.transpose(key, axes=tuple(axes))

        scores = query @ key_T
        # Keep the scale scalar on the same device as the query (CPU or GPU)
        scores = scores * Tensor(
            np.array([self.scale], dtype=np.float32), device=query.device
        )

        if mask is not None:
            scores = scores + mask

        attn_weights = softmax(scores)
        return attn_weights @ value


class MultiHeadAttention(Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.W_o = Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None) -> Tensor:
        """
        Args:
            query, key, value: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len = query.shape[:2]

        # Linear projections
        q = self.W_q(query)  # (batch, seq_len, d_model)
        k = self.W_k(key)
        v = self.W_v(value)

        # Use ops.reshape and ops.transpose to keep gradients flowing
        q = ops.reshape(q, newshape=(batch_size, seq_len, self.num_heads, self.d_k))
        k = ops.reshape(k, newshape=(batch_size, seq_len, self.num_heads, self.d_k))
        v = ops.reshape(v, newshape=(batch_size, seq_len, self.num_heads, self.d_k))

        # Transpose: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        q = ops.transpose(q, axes=(0, 2, 1, 3))
        k = ops.transpose(k, axes=(0, 2, 1, 3))
        v = ops.transpose(v, axes=(0, 2, 1, 3))

        # Apply attention (scaled independently for each head)
        attn_output = self.attention(q, k, v, mask)

        # Transpose back: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        attn_output = ops.transpose(attn_output, axes=(0, 2, 1, 3))

        # Reshape: (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = ops.reshape(
            attn_output, newshape=(batch_size, seq_len, self.d_model)
        )

        # Final linear projection
        output = self.W_o(attn_output)
        return output


class SelfAttention(Module):
    """Self-attention layer (query, key, value all from same source)."""

    def __init__(self, d_model: int, num_heads: int = 1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        return self.mha(x, x, x, mask)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification."""

    def forward(self, logits: Tensor, targets) -> Tensor:
        shape = logits.shape
        if len(shape) > 2:
            B, T, V = shape[0], shape[1], shape[2]
            logits_flat = ops.reshape(logits, newshape=(B * T, V))
        else:
            V = shape[1]
            logits_flat = logits

        targets_np = (
            as_numpy(targets.data if isinstance(targets, Tensor) else targets)
            .flatten()
            .astype(int)
        )

        probs = ops.softmax(logits_flat)
        eps_tensor = Tensor(np.array([1e-8]))  # float64
        log_probs = ops.log(probs + eps_tensor)

        one_hot = np.eye(V)[targets_np]  # float64
        t_one_hot = Tensor(one_hot)

        selected = log_probs * t_one_hot
        summed = ops.sum(selected, axis=-1)
        return -ops.mean(summed)


# ============================================================================
# UTILITY LAYERS
# ============================================================================


class Flatten(Module):
    """Flattens a tensor along specified dimensions."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape  # Tensor.shape is already a tuple
        ndim = len(shape)
        start = self.start_dim % ndim
        end = self.end_dim % ndim

        if start > end:
            raise ValueError(
                f"Invalid range [{start}, {end}] for flattening tensor of ndim {ndim}."
            )

        new_shape = shape[:start] + (-1,) + shape[end + 1 :]
        # ops.reshape registers the node in VJP_RULES, so gradients flow properly.
        return ops.reshape(x, newshape=new_shape)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Base classes
    "Module",
    "Sequential",
    # Linear layers
    "Linear",
    # Convolutional layers
    "Conv2d",
    # Pooling layers
    "MaxPool2d",
    "AvgPool2d",
    # Activation layers
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    # Normalization layers
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    "Embedding",
    # Regularization layers
    "Dropout",
    # Attention layers
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "SelfAttention",
    "CrossEntropyLoss",
    # Utility layers
    "Flatten",
]
