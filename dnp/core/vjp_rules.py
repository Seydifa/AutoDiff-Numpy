"""
dnp/core/vjp_rules.py
=====================
Vector-Jacobian Product (VJP) rules for all supported operations,
plus activation functions and convolution helpers.

This is the single source of truth — previously split across dnp/ops/vjp_rules.py.
"""

# Third-party libraries
import numpy as np
from scipy.signal import convolve2d

EPSILON = 1e-8

# ---------------------------------------------------------------------------
# Broadcasting utility
# ---------------------------------------------------------------------------


def unbroadcast(grad, target_shape):
    """Sum gradient axes that were broadcast-expanded, restoring target_shape."""
    if grad.shape == target_shape:
        return grad
    ndims_added = grad.ndim - len(target_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# ---------------------------------------------------------------------------
# Activation / helper functions (also serve as forward-pass callables)
# ---------------------------------------------------------------------------


def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def relu(x):
    return np.maximum(0.0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))


def softplus(x):
    x_clipped = np.clip(x, -500, 500)
    return np.log(1.0 + np.exp(x_clipped))


def swish(x):
    return x * sigmoid(x)


def gelu(x):
    const1 = np.sqrt(2.0 / np.pi)
    const2 = 0.044715
    return 0.5 * x * (1.0 + np.tanh(const1 * (x + const2 * np.power(x, 3))))


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Pooling operations (vectorized)
# ---------------------------------------------------------------------------


def max_pool2d(x, kernel_size, stride=1, padding=0):
    """
    Vectorized 2D max pooling.

    Args:
        x: (batch, channels, height, width)
        kernel_size: int or tuple (kH, kW)
        stride: int or tuple (sH, sW)
        padding: int
    Returns:
        output: (batch, channels, height_out, width_out)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    batch, channels, H, W = x.shape
    kH, kW = kernel_size
    sH, sW = stride

    # Pad if necessary
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        H, W = x.shape[2:]

    # Calculate output shape
    H_out = (H - kH) // sH + 1
    W_out = (W - kW) // sW + 1

    # Use stride tricks for vectorization
    strides = x.strides
    shape = (
        batch,
        channels,
        H_out,
        W_out,
        kH,
        kW,
    )
    strides = (
        strides[0],
        strides[1],
        sH * strides[2],
        sW * strides[3],
        strides[2],
        strides[3],
    )

    x_windowed = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.max(x_windowed, axis=(4, 5))


def avg_pool2d(x, kernel_size, stride=1, padding=0):
    """
    Vectorized 2D average pooling.

    Args:
        x: (batch, channels, height, width)
        kernel_size: int or tuple (kH, kW)
        stride: int or tuple (sH, sW)
        padding: int
    Returns:
        output: (batch, channels, height_out, width_out)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    batch, channels, H, W = x.shape
    kH, kW = kernel_size
    sH, sW = stride

    # Pad if necessary
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        H, W = x.shape[2:]

    # Calculate output shape
    H_out = (H - kH) // sH + 1
    W_out = (W - kW) // sW + 1

    # Use stride tricks for vectorization
    strides = x.strides
    shape = (
        batch,
        channels,
        H_out,
        W_out,
        kH,
        kW,
    )
    strides = (
        strides[0],
        strides[1],
        sH * strides[2],
        sW * strides[3],
        strides[2],
        strides[3],
    )

    x_windowed = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.mean(x_windowed, axis=(4, 5))


def dropout(x, p=0.5, training=True):
    """
    Dropout regularization.

    Args:
        x: input tensor
        p: dropout probability
        training: if True, apply dropout; if False, return x unchanged
    Returns:
        output: x with dropout applied (scaled by 1/(1-p))
    """
    if not training or p == 0:
        return x
    mask = np.random.binomial(1, 1 - p, size=x.shape)
    return x * mask / (1 - p)


def batch_norm(x, weight, bias, mean, var, eps=1e-5):
    """
    Batch normalization: (x - mean) / sqrt(var + eps) * weight + bias

    Args:
        x: input tensor
        weight: scale parameter (gamma)
        bias: shift parameter (beta)
        mean: batch mean
        var: batch variance
        eps: epsilon for numerical stability
    Returns:
        output: normalized tensor
    """
    x_norm = (x - mean) / np.sqrt(var + eps)
    return weight * x_norm + bias


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------


def conv2d(x, w, mode="valid"):
    """2D convolution using scipy.signal.convolve2d."""
    return convolve2d(x, w, mode=mode)


def rot180(w):
    """Rotate kernel 180° (equivalent to flipping both axes)."""
    return np.rot90(w, 2)


def conv2d_full(g, w):
    """Full convolution used for gradient calculation."""
    return convolve2d(g, w, mode="full")


# ---------------------------------------------------------------------------
# VJP Rules dictionary
# ---------------------------------------------------------------------------

VJP_RULES = {
    # ======================================================================
    # 1. BINARY OPERATIONS  (broadcasting handled via unbroadcast)
    # ======================================================================
    np.add: lambda g, x, y: (unbroadcast(g, x.shape), unbroadcast(g, y.shape)),
    np.subtract: lambda g, x, y: (unbroadcast(g, x.shape), unbroadcast(-g, y.shape)),
    np.multiply: lambda g, x, y: (
        unbroadcast(g * y, x.shape),
        unbroadcast(g * x, y.shape),
    ),
    np.divide: lambda g, x, y: (
        unbroadcast(g / y, x.shape),
        unbroadcast(-g * x / (y**2 + EPSILON), y.shape),
    ),
    np.power: lambda g, x, y: (
        unbroadcast(g * y * np.power(x, y - 1), x.shape),
        unbroadcast(g * np.power(x, y) * np.log(x + EPSILON), y.shape),
    ),
    np.maximum: lambda g, x, y: (
        unbroadcast(g * (x >= y), x.shape),
        unbroadcast(g * (x < y), y.shape),
    ),
    np.minimum: lambda g, x, y: (
        unbroadcast(g * (x <= y), x.shape),
        unbroadcast(g * (x > y), y.shape),
    ),
    # ======================================================================
    # 2. UNARY OPERATIONS
    # ======================================================================
    np.negative: lambda g, x: (-g,),
    np.square: lambda g, x: (g * 2 * x,),
    np.sqrt: lambda g, x: (g / (2 * np.sqrt(x) + EPSILON),),
    np.exp: lambda g, x: (g * np.exp(x),),
    np.log: lambda g, x: (g / (x + EPSILON),),
    np.log1p: lambda g, x: (g / (x + 1.0 + EPSILON),),
    np.expm1: lambda g, x: (g * np.exp(x),),
    np.abs: lambda g, x: (g * np.sign(x),),
    np.sign: lambda g, x: (np.zeros_like(x),),  # 0 a.e.
    # Trigonometry & Hyperbolic
    np.sin: lambda g, x: (g * np.cos(x),),
    np.cos: lambda g, x: (g * -np.sin(x),),
    np.tan: lambda g, x: (g / (np.cos(x) ** 2 + EPSILON),),
    np.sinh: lambda g, x: (g * np.cosh(x),),
    np.cosh: lambda g, x: (g * np.sinh(x),),
    np.tanh: lambda g, x: (g * (1 - np.tanh(x) ** 2),),
    # Rounding (gradient is 0 a.e.)
    np.floor: lambda g, x: (np.zeros_like(x),),
    np.ceil: lambda g, x: (np.zeros_like(x),),
    np.round: lambda g, x: (np.zeros_like(x),),
    # ======================================================================
    # 3. MATRIX OPERATIONS
    # ======================================================================
    np.matmul: lambda g, x, y: (np.matmul(g, y.T), np.matmul(x.T, g)),
    np.dot: lambda g, x, y: (np.dot(g, y.T), np.dot(x.T, g)),
    # ======================================================================
    # 4. REDUCTIONS  (axis + keepdims aware)
    # ======================================================================
    np.sum: lambda g, x, axis=None, keepdims=False: (
        (g if keepdims else np.expand_dims(g, axis) if axis is not None else g)
        * np.ones_like(x),
    ),
    np.mean: lambda g, x, axis=None, keepdims=False: (
        (
            (g if keepdims else np.expand_dims(g, axis) if axis is not None else g)
            * np.ones_like(x)
        )
        / (np.prod(np.array(x.shape)[axis]) if axis is not None else np.size(x)),
    ),
    np.prod: lambda g, x, axis=None, keepdims=False: (
        (g if keepdims else np.expand_dims(g, axis) if axis is not None else g)
        * (np.prod(x, axis=axis, keepdims=True) / (x + EPSILON)),
    ),
    np.max: lambda g, x, axis=None, keepdims=False: (
        (g if keepdims else np.expand_dims(g, axis) if axis is not None else g)
        * (x == np.max(x, axis=axis, keepdims=True)),
    ),
    np.min: lambda g, x, axis=None, keepdims=False: (
        (g if keepdims else np.expand_dims(g, axis) if axis is not None else g)
        * (x == np.min(x, axis=axis, keepdims=True)),
    ),
    # ======================================================================
    # 5. SHAPE OPERATIONS
    # ======================================================================
    np.reshape: lambda g, x, newshape: (np.reshape(g, x.shape),),
    np.transpose: lambda g, x, axes=None: (
        np.transpose(g, np.argsort(axes)) if axes is not None else g.T,
    ),
    np.expand_dims: lambda g, x, axis: (np.squeeze(g, axis),),
    np.squeeze: lambda g, x, axis=None: (
        np.expand_dims(g, axis) if axis is not None else np.reshape(g, x.shape),
    ),
    # ======================================================================
    # 6. CUSTOM / NN OPS
    # ======================================================================
    conv2d: lambda g, x, w, mode="valid": (
        convolve2d(
            g,
            rot180(w),
            mode="full" if mode == "valid" else "same" if mode == "same" else "valid",
        ),
        convolve2d(
            x
            if mode == "valid"
            else np.pad(x, [(w.shape[0] // 2,) * 2, (w.shape[1] // 2,) * 2])
            if mode == "same"
            else np.pad(x, [(w.shape[0] - 1,) * 2, (w.shape[1] - 1,) * 2]),
            g,
            mode="valid",
        ),
    ),
    sigmoid: lambda g, x: (g * sigmoid(x) * (1.0 - sigmoid(x)),),
    relu: lambda g, x: (g * (x > 0).astype(x.dtype),),
    leaky_relu: lambda g, x: (g * np.where(x > 0, 1.0, 0.01),),
    elu: lambda g, x: (g * np.where(x > 0, 1.0, 1.0 * np.exp(x)),),
    softplus: lambda g, x: (g * sigmoid(x),),
    swish: lambda g, x: (g * (swish(x) + sigmoid(x) * (1.0 - swish(x))),),
    gelu: lambda g, x: (
        g
        * (
            0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
            + 0.5
            * x
            * (1.0 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)) ** 2)
            * np.sqrt(2 / np.pi)
            * (1.0 + 3 * 0.044715 * x**2)
        ),
    ),
    softmax: lambda g, x: (
        g * softmax(x) - softmax(x) * np.sum(g * softmax(x), axis=-1, keepdims=True),
    ),
    # ======================================================================
    # 7. POOLING OPERATIONS
    # ======================================================================
    max_pool2d: lambda g, x, kernel_size, stride=1, padding=0: (
        # For max pooling: gradient flows only to the max element
        _max_pool2d_backward(g, x, kernel_size, stride, padding),
    ),
    avg_pool2d: lambda g, x, kernel_size, stride=1, padding=0: (
        # For avg pooling: gradient is uniform across the pooled region
        _avg_pool2d_backward(g, x, kernel_size, stride, padding),
    ),
    # ======================================================================
    # 8. REGULARIZATION OPERATIONS
    # ======================================================================
    dropout: lambda g, x, p=0.5, training=True: (
        (g / (1 - p),) if training else (g,)
    ),  # Gradient scaled by 1/(1-p) during training
    # ======================================================================
    # 9. NORMALIZATION OPERATIONS
    # ======================================================================
    batch_norm: lambda g, x, weight, bias, mean, var, eps=1e-5: (
        _batch_norm_backward(g, x, weight, bias, mean, var, eps),
    ),
}


# ---------------------------------------------------------------------------
# Backward pass helpers for pooling and normalization
# ---------------------------------------------------------------------------


def _max_pool2d_backward(g, x, kernel_size, stride=1, padding=0):
    """Backward pass for max pooling."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    batch, channels, H, W = x.shape
    kH, kW = kernel_size
    sH, sW = stride

    # Pad input if necessary
    x_padded = x
    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    # Calculate output shape
    H_padded = x_padded.shape[2]
    W_padded = x_padded.shape[3]
    H_out = (H_padded - kH) // sH + 1
    W_out = (W_padded - kW) // sW + 1

    # Initialize gradient for input
    grad_x = np.zeros_like(x_padded)

    # Backward pass: distribute gradient to max elements
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * sH
            h_end = h_start + kH
            w_start = j * sW
            w_end = w_start + kW

            window = x_padded[:, :, h_start:h_end, w_start:w_end]
            max_vals = np.max(window, axis=(2, 3), keepdims=True)
            mask = (window == max_vals).astype(x.dtype)
            grad_x[:, :, h_start:h_end, w_start:w_end] += (
                mask * g[:, :, i : i + 1, j : j + 1]
            )

    if padding > 0:
        grad_x = grad_x[:, :, padding:-padding, padding:-padding]

    return grad_x


def _avg_pool2d_backward(g, x, kernel_size, stride=1, padding=0):
    """Backward pass for average pooling."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    batch, channels, H, W = x.shape
    kH, kW = kernel_size
    sH, sW = stride
    pool_size = kH * kW

    # Pad input if necessary
    x_padded = x
    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    # Calculate output shape
    H_padded = x_padded.shape[2]
    W_padded = x_padded.shape[3]
    H_out = (H_padded - kH) // sH + 1
    W_out = (W_padded - kW) // sW + 1

    # Initialize gradient for input
    grad_x = np.zeros_like(x_padded)

    # Backward pass: distribute gradient uniformly to pooled region
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * sH
            h_end = h_start + kH
            w_start = j * sW
            w_end = w_start + kW

            grad_x[:, :, h_start:h_end, w_start:w_end] += (
                g[:, :, i : i + 1, j : j + 1] / pool_size
            )

    if padding > 0:
        grad_x = grad_x[:, :, padding:-padding, padding:-padding]

    return grad_x


def _batch_norm_backward(g, x, weight, bias, mean, var, eps=1e-5):
    """
    Backward pass for batch normalization.
    Returns gradients for x only (weight and bias gradients computed separately).
    """
    # x_norm = (x - mean) / sqrt(var + eps)
    x_norm = (x - mean) / np.sqrt(var + eps)

    # dL/dx_norm = dL/dy * weight
    grad_x_norm = g * weight

    # dL/dx = dL/dx_norm * d(x_norm)/dx
    # d(x_norm)/dx = 1 / sqrt(var + eps) - x_norm / (var + eps) * d(var)/dx - 1 / sqrt(var + eps) * d(mean)/dx
    N = x.shape[0]
    grad_x = (
        grad_x_norm / np.sqrt(var + eps)
        - (grad_x_norm * x_norm) * (1 / (var + eps)) * (x - mean) / N
        - np.mean(grad_x_norm, axis=0, keepdims=True) / np.sqrt(var + eps)
    )

    return grad_x
