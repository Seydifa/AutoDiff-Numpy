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

# Local imports: Core tensor and operations
from .tensor import Tensor
from . import ops
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
            p.grad = np.zeros_like(p)

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
        y = matmul(x, self.W)
        if self.use_bias:
            y = add(y, self.b)
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
        """Forward pass: (batch, in_channels, H, W) -> (batch, out_channels, H', W')."""
        from scipy.signal import convolve2d

        batch_size, in_channels, H, W = x.shape
        out_channels, _, kH, kW = self.W.shape
        stride_h, stride_w = self.stride
        pad = self.padding

        # Handle string padding modes
        if isinstance(pad, str):
            if pad.lower() == "same":
                # For "same" mode with stride=1, compute padding to preserve spatial shape
                # For stride > 1, output size is ceil(H / stride)
                pad_h = max(0, (kH - 1) // 2)
                pad_w = max(0, (kW - 1) // 2)
            elif pad.lower() == "valid":
                pad_h, pad_w = 0, 0
            else:
                raise ValueError(f"Unknown padding mode: {pad}")
        else:
            pad_h = pad_w = pad

        # Apply padding
        if pad_h > 0 or pad_w > 0:
            x_padded = np.pad(
                x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
            )
        else:
            x_padded = x

        # Compute output spatial dimensions
        H_out = (x_padded.shape[2] - kH) // stride_h + 1
        W_out = (x_padded.shape[3] - kW) // stride_w + 1

        # Initialize output tensor
        y = np.zeros((batch_size, out_channels, H_out, W_out), dtype=x.dtype)

        # Perform convolution for each sample in batch and each output channel
        for b in range(batch_size):
            for oc in range(out_channels):
                # Convolution for this output channel across input channels
                for ic in range(in_channels):
                    # Slice the input for this batch and input channel
                    x_ic = x_padded[b, ic, :, :]
                    w_ic = self.W[oc, ic, :, :]

                    # Perform 2D convolution
                    conv_result = convolve2d(x_ic, w_ic, mode="valid")

                    # Apply stride and accumulate
                    y[b, oc, :, :] += conv_result[::stride_h, ::stride_w]

        # Apply bias if present
        if self.use_bias:
            # Reshape bias to (1, out_channels, 1, 1) for broadcasting
            bias_reshaped = self.b.reshape(1, out_channels, 1, 1)
            y = add(y, bias_reshaped)
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

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using batch norm operation."""
        if self.training:
            mean = np.mean(np.asarray(x), axis=0)
            var = np.var(np.asarray(x), axis=0)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Use batch_norm operation with proper shape broadcasting
        return batch_norm(x, self.weight, self.bias, mean, var, self.eps)


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

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using batch norm operation."""
        x_np = np.asarray(x)
        if self.training:
            mean = np.mean(x_np, axis=(0, 2, 3))
            var = np.var(x_np, axis=(0, 2, 3))

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
        shape_for_norm = (1, self.num_features, 1, 1)
        mean_reshaped = mean.reshape(shape_for_norm)
        var_reshaped = var.reshape(shape_for_norm)
        weight_reshaped = np.asarray(self.weight).reshape(shape_for_norm)
        bias_reshaped = np.asarray(self.bias).reshape(shape_for_norm)

        # Use batch_norm operation
        return batch_norm(
            x,
            Tensor(weight_reshaped),
            Tensor(bias_reshaped),
            mean_reshaped,
            var_reshaped,
            self.eps,
        )


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
        # Calculate mean along last axis
        mean = ops.mean(x, axis=-1, keepdims=True)
        diff = ops.subtract(x, mean)
        sq_diff = ops.square(diff)
        var = ops.mean(sq_diff, axis=-1, keepdims=True)
        std = ops.sqrt(ops.add(var, self.eps))
        x_norm = ops.divide(diff, std)
        out = ops.multiply(x_norm, self.gamma)
        if self.use_bias:
            out = ops.add(out, self.beta)
        return out


class Embedding(Module):
    """Embedding layer constructed with one-hot encoding for autograd compatibility."""
    def __init__(self, num_embeddings, embedding_dim, name="Embedding"):
        super().__init__()
        self.lin = Linear(num_embeddings, embedding_dim, bias=False, name=name)
        self.num_embeddings = num_embeddings
        
    def forward(self, x_idx):
        x_idx_np = np.asarray(x_idx) if hasattr(x_idx, 'data') else x_idx
        one_hot = np.eye(self.num_embeddings)[x_idx_np]
        return self.lin(Tensor(one_hot))


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
        # Compute attention scores by transposing last 2 dims of key
        query_np = np.asarray(query)
        key_np = np.asarray(key)
        value_np = np.asarray(value)

        # Transpose last 2 dimensions: (..., seq_len_k, d_k) -> (..., d_k, seq_len_k)
        key_transposed = np.swapaxes(key_np, -2, -1)
        scores = matmul(query, Tensor(key_transposed))
        scores = scores * self.scale

        if mask is not None:
            scores = scores + mask

        # Apply softmax
        attn_weights = softmax(scores)

        # Apply attention to values
        output = matmul(attn_weights, value)
        return output


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
        batch_size = np.asarray(query).shape[0]
        seq_len = np.asarray(query).shape[1]

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
        attn_output = ops.reshape(attn_output, newshape=(batch_size, seq_len, self.d_model))

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
        shape = np.asarray(logits).shape
        if len(shape) > 2:
            B, T, V = shape[0], shape[1], shape[2]
            logits_flat = ops.reshape(logits, newshape=(B * T, V))
        else:
            V = shape[1]
            logits_flat = logits
            
        targets_flat = np.asarray(targets).flatten()
            
        probs = ops.softmax(logits_flat)
        eps_tensor = Tensor(np.array([1e-8], dtype=np.float32))
        log_probs = ops.log(ops.add(probs, eps_tensor))
        
        one_hot = np.eye(V, dtype=np.float32)[targets_flat]
        t_one_hot = Tensor(one_hot)
        
        selected = ops.multiply(log_probs, t_one_hot)
        summed = ops.sum(selected, axis=-1)
        loss = ops.negative(ops.mean(summed))
        return loss

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
        shape = np.asarray(x).shape
        ndim = len(shape)
        start = self.start_dim % ndim
        end = self.end_dim % ndim

        if start > end:
            raise ValueError(
                f"Invalid range [{start}, {end}] for flattening tensor of ndim {ndim}."
            )

        new_shape = (
            shape[:start] + (-1,) + shape[end + 1 :]
        )  # -1 replaces flattened dims

        return Tensor(
            np.asarray(x).reshape(new_shape),
            parents=[x],
            op_func=lambda: None,
            name="Flatten",
        )


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
