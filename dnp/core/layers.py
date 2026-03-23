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
from .backend import get_xp, as_numpy, get_dtype, safe_eps

# Local imports: Core tensor and operations
from .tensor import Tensor
from . import ops
from .vjp_rules import _conv2d_forward_kernel
from .ops import (
    matmul,
    add,
    relu,
    leaky_relu,
    elu,
    softplus,
    swish,
    gelu,
    sigmoid,
    tanh,
    softmax,
    max_pool2d,
    avg_pool2d,
    dropout,
    batch_norm,
    gather,
)

# ============================================================================
# BASE CLASSES
# ============================================================================


def _apply_initializer(name: str, shape, dtype) -> np.ndarray:
    """Resolve a built-in initializer name to an initial weight ndarray."""
    fan_in = shape[0] if len(shape) >= 1 else 1
    fan_out = shape[1] if len(shape) >= 2 else 1
    limit: float
    if name == "glorot_uniform":
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape).astype(dtype)
    if name == "glorot_normal":
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return (np.random.randn(*shape) * std).astype(dtype)
    if name == "he_normal":
        std = np.sqrt(2.0 / fan_in)
        return (np.random.randn(*shape) * std).astype(dtype)
    if name == "ones":
        return np.ones(shape, dtype=dtype)
    if name == "zeros":
        return np.zeros(shape, dtype=dtype)
    if name == "uniform":
        return np.random.uniform(-0.05, 0.05, shape).astype(dtype)
    raise ValueError(
        f"Unknown initializer '{name}'. "
        "Choose from 'glorot_uniform', 'glorot_normal', 'he_normal', 'ones', 'zeros', 'uniform'."
    )


class Module:
    """Base class for all neural network modules.

    v3 additions
    ------------
    * ``add_weight(name, shape, initializer, trainable, **kwargs)`` —
      structured weight registration inspired by Keras.  Weights registered
      through this API are accessible both as ``self.<name>`` and through
      ``self.parameters()`` (if trainable).  Non-trainable weights are stored
      in ``_buffers`` and not returned by ``parameters()``.
    * ``_buffers`` dict for non-trainable persistent state (e.g. running stats).
    * Each instance gets a unique ``_instance_name`` (e.g. ``'Linear_0'``,
      ``'Linear_1'``) so weights from different layers of the same type
      never share the same tensor name in the computation graph.
    """

    # Class-level counter: {class_name: count} — gives each instance a unique index.
    _instance_counters: dict = {}

    def __init__(self):
        # Use __dict__ directly to avoid triggering __setattr__ recursively.
        self.__dict__["_parameters"] = {}  # trainable weights
        self.__dict__["_nontrainable"] = {}  # non-trainable tensors
        self.__dict__["_buffers"] = {}  # plain arrays (not Tensors)
        self.__dict__["_modules"] = {}
        self.__dict__["training"] = True
        # Assign a unique name (e.g. "Linear_0", "Linear_1") per instance so
        # weights registered via add_weight() get distinct tensor names in the
        # computation graph even when multiple layers share the same class.
        cls_name = self.__class__.__name__
        idx = Module._instance_counters.get(cls_name, -1) + 1
        Module._instance_counters[cls_name] = idx
        self.__dict__["_instance_name"] = f"{cls_name}_{idx}"

    # ------------------------------------------------------------------
    # weight registration API
    # ------------------------------------------------------------------

    def add_weight(
        self,
        name: str,
        shape,
        initializer="glorot_uniform",
        trainable: bool = True,
        dtype=None,
        **kwargs,
    ) -> "Tensor":
        """Create and register a weight tensor.

        Parameters
        ----------
        name : str
            Attribute name under which the weight is stored (``self.<name>``).
        shape : tuple
            Shape of the weight tensor.
        initializer : str or callable, default='glorot_uniform'
            How to initialise the weight data.  Built-in options:
            ``'glorot_uniform'``, ``'glorot_normal'``, ``'he_normal'``,
            ``'ones'``, ``'zeros'``, ``'uniform'``, or any callable
            that accepts *shape* and returns an ndarray.
        trainable : bool, default=True
            If ``True`` the weight is returned by ``parameters()`` and
            receives gradients during ``backward()``.  If ``False``
            it is stored in ``_nontrainable`` (accessible as
            ``self.<name>`` but skipped by optimizers).
        dtype : numpy dtype or None
            Override the global default dtype.

        Returns
        -------
        Tensor
            The newly created weight tensor (also set as ``self.<name>``).
        """
        from .backend import get_dtype

        _dtype = dtype or get_dtype()

        if callable(initializer) and not isinstance(initializer, str):
            data = initializer(shape).astype(_dtype)
        else:
            data = _apply_initializer(initializer, shape, _dtype)

        w = Tensor(data, name=f"{self._instance_name}.{name}")
        object.__setattr__(self, name, w)
        if trainable:
            self._parameters[name] = w
        else:
            self._nontrainable[name] = w
        return w

    # ------------------------------------------------------------------
    # attribute routing
    # ------------------------------------------------------------------

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            if name in self._parameters:
                # Keep _parameters in sync when an existing weight is reassigned.
                self._parameters[name] = value
            elif name in self._nontrainable:
                self._nontrainable[name] = value
            else:
                # New tensor — register as a trainable parameter.
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

        self.W = self.add_weight(
            "W",
            shape=(in_features, out_features),
            initializer=lambda s: np.random.randn(*s) * std,
        )
        if bias:
            self.b = self.add_weight("b", shape=(out_features,), initializer="zeros")
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

        self.W = self.add_weight(
            "W",
            shape=(out_channels, in_channels, kH, kW),
            initializer=lambda s: np.random.randn(*s) * std,
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

        # Handle string padding modes
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


class LeakyReLU(_ActivationModule):
    """Leaky ReLU activation: max(alpha*x, x)."""

    def __init__(self, alpha: float = 0.01):
        super().__init__(
            lambda x: leaky_relu(x, alpha=alpha), f"LeakyReLU(alpha={alpha})"
        )
        self.alpha = alpha


class ELU(_ActivationModule):
    """Exponential Linear Unit activation."""

    def __init__(self, alpha: float = 1.0):
        super().__init__(lambda x: elu(x, alpha=alpha), f"ELU(alpha={alpha})")
        self.alpha = alpha


class Softplus(_ActivationModule):
    """Softplus activation: log(1 + exp(x))."""

    def __init__(self):
        super().__init__(softplus, "Softplus")


class Swish(_ActivationModule):
    """Swish / SiLU activation: x * sigmoid(x)."""

    def __init__(self):
        super().__init__(swish, "Swish")


class GELU(_ActivationModule):
    """Gaussian Error Linear Unit activation."""

    def __init__(self):
        super().__init__(gelu, "GELU")


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

        self.weight = self.add_weight(
            "weight", shape=(num_features,), initializer="ones"
        )  # γ
        self.bias = self.add_weight(
            "bias", shape=(num_features,), initializer="zeros"
        )  # β

        self.running_mean = np.zeros(num_features, dtype=get_dtype())
        self.running_var = np.ones(num_features, dtype=get_dtype())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using batch norm operation."""
        xp = get_xp(x.data)
        if self.training:
            mean = xp.mean(x.data, axis=0)
            var = xp.var(x.data, axis=0)
            # Keep running stats on the same device as x; migrate once on device switch.
            if get_xp(self.running_mean) is not xp:
                self.running_mean = xp.asarray(self.running_mean)
                self.running_var = xp.asarray(self.running_var)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
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

        self.weight = self.add_weight(
            "weight", shape=(num_features,), initializer="ones"
        )  # γ
        self.bias = self.add_weight(
            "bias", shape=(num_features,), initializer="zeros"
        )  # β

        self.running_mean = np.zeros(num_features, dtype=get_dtype())
        self.running_var = np.ones(num_features, dtype=get_dtype())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using batch norm operation."""
        xp = get_xp(x.data)
        if self.training:
            mean = xp.mean(x.data, axis=(0, 2, 3))
            var = xp.var(x.data, axis=(0, 2, 3))
            # Keep running stats on the same device as x; migrate once on device switch.
            if get_xp(self.running_mean) is not xp:
                self.running_mean = xp.asarray(self.running_mean)
                self.running_var = xp.asarray(self.running_var)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
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
        self.gamma = self.add_weight("gamma", shape=(ndim,), initializer="ones")
        self.use_bias = bias
        if bias:
            self.beta = self.add_weight("beta", shape=(ndim,), initializer="zeros")

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.mean(x, axis=-1, keepdims=True)
        diff = x - mean
        var = ops.mean(ops.square(diff), axis=-1, keepdims=True)
        # Use a plain Python float for eps — not a Tensor leaf node, so it
        # does not add a spurious node to the session graph on each forward call.
        std = ops.sqrt(var + float(self.eps))
        x_norm = diff / std
        out = x_norm * self.gamma
        if self.use_bias:
            out = out + self.beta
        return out


class Embedding(Module):
    """Embedding lookup layer with efficient direct-index slicing.

    Uses ``ops.gather`` — a dedicated ``_GatherOps`` instance whose ``vpj()``
    implements the scatter-add backward.  The weight matrix is registered via
    ``add_weight()`` so it participates in ``parameters()``, ``zero_grad()``,
    and ``named_parameters()`` like every other trainable weight.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, name: str = "Embedding"
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Use add_weight so the embedding matrix is a first-class Module weight.
        self.W = self.add_weight(
            "W",
            shape=(num_embeddings, embedding_dim),
            # Scale initialisation: std = 1 / sqrt(num_embeddings)
            initializer=lambda shape: np.random.randn(*shape) * np.sqrt(1.0 / shape[0]),
        )

    def forward(self, x_idx):
        """x_idx : integer array / Tensor of shape (seq_len,) or (batch, seq_len)."""
        x_data = x_idx.data if isinstance(x_idx, Tensor) else x_idx
        # astype(int) on the native device — cupy accepts xp int arrays for fancy indexing.
        return gather(self.W, x_data.astype(int))


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
        self.scale = float(1.0 / np.sqrt(d_k))

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
        ndim = len(key.shape)
        axes = list(range(ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        key_T = ops.transpose(key, axes=tuple(axes))

        scores = query @ key_T
        # Multiply by a Python float — avoids creating a leaf Tensor node on
        # every forward call, which would grow the session graph indefinitely.
        scores = scores * self.scale

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
            logits_flat = logits

        _t = targets.data if isinstance(targets, Tensor) else targets
        xp = get_xp(logits_flat.data)
        targets_i = xp.asarray(_t).flatten().astype(xp.int64)

        return ops.sparse_cce_with_logits_loss(logits_flat, Tensor(targets_i))


# ============================================================================
# UTILITY LAYERS
# ============================================================================


class MSELoss(Module):
    """Mean Squared Error loss.

    .. math::
        \\mathcal{L} = \\frac{1}{N} \\sum_i (y_{\\text{pred},i} - y_{\\text{true},i})^2

    Parameters
    ----------
    reduction : ``"mean"`` | ``"sum"`` | ``"none"``
        Specifies the reduction to apply over the batch dimension.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true) -> Tensor:
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        if self.reduction == "mean":
            return ops.mse_loss(y_pred, y_true)
        diff = y_pred - y_true
        sq = ops.square(diff)
        if self.reduction == "sum":
            return ops.sum(sq)
        return sq  # 'none'


class BCELoss(Module):
    """Binary Cross-Entropy loss (expects probabilities, not logits).

    .. math::
        \\mathcal{L} = -\\frac{1}{N}\\sum_i \\bigl[
            y_i \\log(\\hat{y}_i + \\epsilon)
            + (1 - y_i) \\log(1 - \\hat{y}_i + \\epsilon)
        \\bigr]

    Parameters
    ----------
    reduction : ``"mean"`` | ``"sum"`` | ``"none"``
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true) -> Tensor:
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        if self.reduction == "mean":
            return ops.bce_loss(y_pred, y_true)
        # fall back to composed ops for sum / none
        eps = safe_eps(y_pred.data)
        pos = y_true * ops.log(y_pred + eps)
        neg = (1.0 - y_true) * ops.log(1.0 - y_pred + eps)
        loss = -(pos + neg)
        if self.reduction == "sum":
            return ops.sum(loss)
        return loss  # 'none'


class BCEWithLogitsLoss(Module):
    """Binary Cross-Entropy with logits (numerically stable sigmoid + BCE).

    Applies :math:`\\sigma(x)` internally, so pass raw logits — do **not**
    apply sigmoid beforehand.

    .. math::
        \\mathcal{L} = -\\frac{1}{N}\\sum_i \\bigl[
            y_i \\log(\\sigma(x_i))
            + (1-y_i) \\log(1-\\sigma(x_i))
        \\bigr]

    which simplifies to:

    .. math::
        \\mathcal{L} = \\frac{1}{N}\\sum_i \\bigl[
            \\max(x_i, 0) - x_i y_i + \\log(1 + e^{-|x_i|})
        \\bigr]

    Parameters
    ----------
    reduction : ``"mean"`` | ``"sum"`` | ``"none"``
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(self, logits: Tensor, y_true) -> Tensor:
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        if self.reduction == "mean":
            return ops.bce_with_logits_loss(logits, y_true)
        # fall back to composed ops for sum / none
        relu_x = ops.relu(logits)
        abs_x = ops.absolute(logits)
        loss = relu_x - logits * y_true + ops.log1p(ops.exp(-abs_x))
        if self.reduction == "sum":
            return ops.sum(loss)
        return loss  # 'none'


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
# TRANSFORMER BUILDING BLOCKS
# ============================================================================


class PositionalEncoding(Module):
    """Sinusoidal positional encoding (non-trainable).

    From "Attention Is All You Need" (Vaswani et al., 2017).  Adds a fixed
    sinusoidal pattern to token embeddings so the model can exploit position.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum sequence length the table is pre-computed for.
    dropout_p : float
        Dropout probability applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout_p: float = 0.1):
        super().__init__()
        self.dropout_p = dropout_p

        # Build (max_len, d_model) sinusoidal table once at init (CPU).
        position = np.arange(max_len)[:, None]  # (max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model), dtype=get_dtype())
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        # Store as a buffer: shape (1, max_len, d_model) for easy broadcasting.
        self._buffers["pe"] = pe[None, :, :]

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model)"""
        xp = get_xp(x.data)
        seq_len = x.shape[1]
        pe = self._buffers["pe"]
        # Lazy device migration — migrate once, then stays on device.
        if get_xp(pe) is not xp:
            pe = xp.asarray(pe)
            self._buffers["pe"] = pe
        pe_slice = Tensor(pe[:, :seq_len, :])
        x = x + pe_slice
        if self.training and self.dropout_p > 0.0:
            x = dropout(x, p=self.dropout_p, training=True)
        return x


class FeedForward(Module):
    """Position-wise Feed-Forward block used in transformer layers.

    Architecture: Linear → activation → Dropout → Linear

    Parameters
    ----------
    d_model : int
        Input / output dimension.
    d_ff : int
        Hidden dimension (typically 4 * d_model).
    activation : str
        One of ``'relu'``, ``'gelu'``, ``'swish'``.
    dropout_p : float
        Dropout probability between the two linear projections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "relu",
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout_p = dropout_p

        _act_map = {"relu": relu, "gelu": gelu, "swish": swish}
        if activation not in _act_map:
            raise ValueError(
                f"Unsupported activation '{activation}'. Choose from {list(_act_map)}."
            )
        self._act_fn = _act_map[activation]

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self._act_fn(x)
        if self.training and self.dropout_p > 0.0:
            x = dropout(x, p=self.dropout_p, training=True)
        return self.fc2(x)


class TransformerEncoderLayer(Module):
    """Single transformer encoder layer.

    Implements the standard pre-norm or post-norm (default) block::

        x = LayerNorm(x + Dropout(MHA(x, x, x, mask)))
        x = LayerNorm(x + FFN(x))

    Parameters
    ----------
    d_model : int
        Model / embedding dimension.
    num_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward hidden dimension.
    dropout_p : float
        Dropout probability for attention output and FFN.
    activation : str
        FFN activation — ``'relu'``, ``'gelu'``, or ``'swish'``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_p: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model, d_ff, activation=activation, dropout_p=dropout_p
        )
        self.norm2 = LayerNorm(d_model)
        self.dropout_p = dropout_p

    def forward(self, x: Tensor, mask=None) -> Tensor:
        # --- Self-attention sublayer ---
        attn_out = self.attn(x, x, x, mask)
        if self.training and self.dropout_p > 0.0:
            attn_out = dropout(attn_out, p=self.dropout_p, training=True)
        x = self.norm1(x + attn_out)
        # --- Feed-forward sublayer ---
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


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
    "LeakyReLU",
    "ELU",
    "Softplus",
    "Swish",
    "GELU",
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
    # Transformer building blocks
    "PositionalEncoding",
    "FeedForward",
    "TransformerEncoderLayer",
    # Loss functions
    "CrossEntropyLoss",
    "MSELoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    # Utility layers
    "Flatten",
]
