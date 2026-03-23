"""
dnp/core/vjp_rules.py
=====================
Vector-Jacobian Product (VJP) rules for all supported operations,
plus activation functions and convolution helpers.

This is the single source of truth — previously split across dnp/ops/vjp_rules.py.

All VJP rules are registered via the ``@vjp_rule`` decorator, which writes
entries into the central ``VJP_RULES`` dict.  To add a new differentiable
operation, define a forward function and decorate its backward with::

    @vjp_rule(func=my_forward_fn)
    def _vjp_my_op(g, *args, **kwargs):
        ...
        return (grad_arg0, grad_arg1, ...)
"""

# Third-party libraries
import numpy as np

from .backend import get_xp, as_numpy, safe_eps

EPSILON = 1e-8

# Precomputed constant for GELU — avoids recomputing sqrt on every call.
_GELU_COEFF = float(np.sqrt(2.0 / np.pi))  # ≈ 0.7978845608028654

# ---------------------------------------------------------------------------
# VJP_RULES registry
# ---------------------------------------------------------------------------
# Central dict: forward_fn → vjp_fn(g, *forward_args, **forward_kwargs).
# Populated entirely by @vjp_rule below — never edit this dict directly.

VJP_RULES: dict = {}


# ---------------------------------------------------------------------------
# vjp_rule — decorator for ergonomic VJP registration
# ---------------------------------------------------------------------------


def vjp_rule(func):
    """Decorator factory that registers a VJP rule for *func* in ``VJP_RULES``.

    Usage::

        @vjp_rule(func=np.sin)
        def _vjp_sin(g, x):
            return (g * get_xp(x).cos(x),)

    The decorated function is stored under ``VJP_RULES[func]`` and returned
    unchanged so it remains directly callable for testing.
    """

    def decorator(vjp_fn):
        VJP_RULES[func] = vjp_fn
        return vjp_fn

    return decorator


# ---------------------------------------------------------------------------
# Broadcasting / axis-restoration utilities
# ---------------------------------------------------------------------------


def unbroadcast(grad, target_shape):
    """Sum gradient axes that were broadcast-expanded, restoring target_shape."""
    if grad.shape == target_shape:
        return grad
    # Collapse all extra leading dims in one fused reduction.
    ndims_added = grad.ndim - len(target_shape)
    if ndims_added > 0:
        grad = grad.sum(axis=tuple(range(ndims_added)))
    # Collapse all broadcast dims in one fused reduction (keepdims preserves rank).
    axes = tuple(
        i for i, dim in enumerate(target_shape) if dim == 1 and grad.shape[i] > 1
    )
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad


def _restore_reduced_dims(g, x_shape, axis, keepdims):
    """Re-insert the squeezed axes into *g* so it broadcasts to *x_shape*.

    Handles axis=None (full reduction), axis=int, and axis=tuple.
    Works on NumPy < 2.0 which does not accept a tuple axis in expand_dims.
    """
    xp = get_xp(g)
    if keepdims or axis is None:
        return g * xp.ones(x_shape, dtype=g.dtype)
    if isinstance(axis, (list, tuple)):
        result = g
        for ax in sorted(int(a) % len(x_shape) for a in axis):
            result = xp.expand_dims(result, ax)
    else:
        result = xp.expand_dims(g, axis)
    return result * xp.ones(x_shape, dtype=g.dtype)


# ---------------------------------------------------------------------------
# Activation / helper functions (also serve as forward-pass callables)
# ---------------------------------------------------------------------------


def sigmoid(x):
    xp = get_xp(x)
    x_clipped = xp.clip(x, -500, 500)
    return 1.0 / (1.0 + xp.exp(-x_clipped))


def relu(x):
    xp = get_xp(x)
    return xp.maximum(0.0, x)


def leaky_relu(x, alpha=0.01):
    xp = get_xp(x)
    return xp.where(x > 0, x, alpha * x)


def elu(x, alpha=1.0):
    xp = get_xp(x)
    return xp.where(x > 0, x, alpha * (xp.exp(x) - 1.0))


def softplus(x):
    xp = get_xp(x)
    # logaddexp(0, x) = log(1 + exp(x)) computed in a numerically stable way.
    return xp.logaddexp(0.0, x)


def swish(x):
    return x * sigmoid(x)


def gelu(x):
    xp = get_xp(x)
    # Use module-level constant (no sqrt per call) and ** instead of xp.power.
    return 0.5 * x * (1.0 + xp.tanh(_GELU_COEFF * (x + 0.044715 * x**3)))


def softmax(x, axis=-1):
    xp = get_xp(x)
    x_max = xp.max(x, axis=axis, keepdims=True)
    e_x = xp.exp(x - x_max)
    return e_x / xp.sum(e_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Pooling operations (vectorized)
# ---------------------------------------------------------------------------


def max_pool2d(x, kernel_size, stride=1, padding=0):
    xp = get_xp(x)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    if padding > 0:
        x = xp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    batch, channels, H, W = x.shape
    kH, kW = kernel_size
    sH, sW = stride
    H_out = (H - kH) // sH + 1
    W_out = (W - kW) // sW + 1

    strides = x.strides
    shape = (batch, channels, H_out, W_out, kH, kW)
    strides = (
        strides[0],
        strides[1],
        sH * strides[2],
        sW * strides[3],
        strides[2],
        strides[3],
    )

    x_windowed = xp.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return xp.max(x_windowed, axis=(4, 5))


def avg_pool2d(x, kernel_size, stride=1, padding=0):
    xp = get_xp(x)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    if padding > 0:
        x = xp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    batch, channels, H, W = x.shape
    kH, kW = kernel_size
    sH, sW = stride
    H_out = (H - kH) // sH + 1
    W_out = (W - kW) // sW + 1

    strides = x.strides
    shape = (batch, channels, H_out, W_out, kH, kW)
    strides = (
        strides[0],
        strides[1],
        sH * strides[2],
        sW * strides[3],
        strides[2],
        strides[3],
    )

    x_windowed = xp.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return xp.mean(x_windowed, axis=(4, 5))


def dropout(x, p=0.5, training=True):
    """Dropout forward — the binary mask is generated in ``_DropoutOps._raw_call``
    and stored in ``op_kwargs["mask"]`` so ``Tensor.backward()`` can retrieve it.
    """
    if not training or p == 0:
        return x
    xp = get_xp(x)
    mask = (xp.random.uniform(size=x.shape) >= p).astype(x.dtype)
    return x * mask / (1 - p)


def batch_norm(x, weight, bias, mean, var, eps=1e-5):
    xp = get_xp(x)
    x_norm = (x - mean) / xp.sqrt(var + eps)
    return weight * x_norm + bias


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------


def conv2d(x, w, mode="valid"):
    """2D convolution using scipy/cupyx."""
    xp = get_xp(x)
    if xp.__name__ == "cupy":
        try:
            from cupyx.scipy.signal import convolve2d as cp_convolve2d

            return cp_convolve2d(x, w, mode=mode)
        except ImportError:
            from scipy.signal import convolve2d as sp_convolve2d

            res = sp_convolve2d(as_numpy(x), as_numpy(w), mode=mode)
            return xp.asarray(res)
    else:
        from scipy.signal import convolve2d as sp_convolve2d

        return sp_convolve2d(x, w, mode=mode)


def rot180(w):
    xp = get_xp(w)
    return xp.rot90(w, 2)


def conv2d_full(g, w):
    return conv2d(g, w, mode="full")


# ---------------------------------------------------------------------------
# Im2col convolution kernel  (forward callable — also the VJP dict key)
# ---------------------------------------------------------------------------


def _conv2d_forward_kernel(x_padded, W, stride_h, stride_w, H_out, W_out):
    """Im2col + batched matmul convolution — O(B·Cout·Cin·kH·kW·Hout·Wout).

    Replaces the old 6-level Numba loop with a fully vectorized path that
    works transparently on both NumPy (CPU) and CuPy (GPU) arrays.
    """
    xp = get_xp(x_padded)
    batch, in_ch, _, _ = x_padded.shape
    out_ch, _, kH, kW = W.shape

    # Build a strided patch view: (batch, in_ch, kH, kW, H_out, W_out)
    s = x_padded.strides
    col_shape = (batch, in_ch, kH, kW, H_out, W_out)
    col_strides = (s[0], s[1], s[2], s[3], stride_h * s[2], stride_w * s[3])
    cols = xp.lib.stride_tricks.as_strided(
        x_padded, shape=col_shape, strides=col_strides
    )
    cols_2d = xp.ascontiguousarray(cols).reshape(batch, in_ch * kH * kW, H_out * W_out)
    W_2d = W.reshape(out_ch, in_ch * kH * kW)
    return xp.matmul(W_2d, cols_2d).reshape(batch, out_ch, H_out, W_out)


# ---------------------------------------------------------------------------
# NumPy-version-safe reshape wrapper
# ---------------------------------------------------------------------------
# NumPy 2.0 renamed the ``newshape`` keyword to ``shape``.  Passing the new
# shape *positionally* works on every version, so we wrap it here.


def _reshape(a, newshape):
    """Backend-agnostic reshape — stays on whichever device *a* lives on."""
    return get_xp(a).reshape(a, newshape)


# ---------------------------------------------------------------------------
# Variadic forward wrappers — individual Tensor tracking per input
# ---------------------------------------------------------------------------


def _concatenate(*arrays, axis=0):
    """Variadic wrapper: ``_concatenate(a, b, c, axis=0)`` → backend concatenate."""
    xp = get_xp(arrays[0])
    return xp.concatenate(arrays, axis=axis)


def _stack(*arrays, axis=0):
    """Variadic wrapper: ``_stack(a, b, c, axis=0)`` → backend stack."""
    xp = get_xp(arrays[0])
    return xp.stack(arrays, axis=axis)


# ---------------------------------------------------------------------------
# Complex backward helpers (logic too involved for a single expression)
# ---------------------------------------------------------------------------


def _norm_pair(v) -> tuple:
    """Normalise a kernel/stride argument to a ``(int, int)`` pair.

    Accepts an ``int``, a 0-d NumPy/CuPy array (produced by the ``_vjp_args``
    scalar-wrapping path in ``Ops.vpj``), or an already-iterable pair.
    """
    # Unwrap 0-d array first so subsequent checks work on plain Python values.
    if hasattr(v, "ndim") and v.ndim == 0:
        v = int(v)
    if isinstance(v, (int, float)):
        v = int(v)
        return (v, v)
    # Iterable pair (list or tuple of ints / 0-d arrays)
    a, b = v
    return (
        int(a) if hasattr(a, "ndim") else int(a),
        int(b) if hasattr(b, "ndim") else int(b),
    )


def _norm_int(v) -> int:
    """Normalise a scalar argument (int or 0-d array) to a plain Python int."""
    if hasattr(v, "ndim") and v.ndim == 0:
        return int(v)
    return int(v)


def _max_pool2d_backward(g, x, kernel_size, stride=1, padding=0):
    """Vectorized max-pool backward via strided mask + kH*kW scatter-adds.

    Each of the kH*kW iterations is a fully vectorized operation over
    (batch, channels, H_out, W_out) — no Python loops over spatial dims.
    Works on both NumPy (CPU) and CuPy (GPU) without any data transfer.
    """
    xp = get_xp(x)
    kernel_size = _norm_pair(kernel_size)
    stride = _norm_pair(stride)
    padding = _norm_int(padding)
    kH, kW = kernel_size
    sH, sW = stride

    if padding > 0:
        x = xp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    batch, channels, H_pad, W_pad = x.shape
    H_out = (H_pad - kH) // sH + 1
    W_out = (W_pad - kW) // sW + 1

    s = x.strides
    win_shape = (batch, channels, H_out, W_out, kH, kW)
    win_strides = (s[0], s[1], sH * s[2], sW * s[3], s[2], s[3])
    x_win = xp.lib.stride_tricks.as_strided(x, shape=win_shape, strides=win_strides)

    max_val = xp.max(x_win, axis=(4, 5), keepdims=True)
    mask = (x_win == max_val).astype(g.dtype)
    mask /= mask.sum(axis=(4, 5), keepdims=True)

    grad_win = g[:, :, :, :, xp.newaxis, xp.newaxis] * mask

    grad_x = xp.zeros_like(x)
    for kh in range(kH):
        for kw in range(kW):
            grad_x[:, :, kh : kh + H_out * sH : sH, kw : kw + W_out * sW : sW] += (
                grad_win[:, :, :, :, kh, kw]
            )

    if padding > 0:
        grad_x = grad_x[:, :, padding:-padding, padding:-padding]
    return grad_x


def _avg_pool2d_backward(g, x, kernel_size, stride=1, padding=0):
    """Vectorized avg-pool backward via uniform scatter-add.

    Each input element within a window receives g / (kH*kW).  Overlapping
    windows accumulate via kH*kW vectorized additions — no Python loops over
    batch, channel, or spatial dimensions.
    """
    xp = get_xp(x)
    kernel_size = _norm_pair(kernel_size)
    stride = _norm_pair(stride)
    padding = _norm_int(padding)
    kH, kW = kernel_size
    sH, sW = stride

    batch, channels, H, W = x.shape
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    H_out = (H_padded - kH) // sH + 1
    W_out = (W_padded - kW) // sW + 1

    g_scaled = g / (kH * kW)
    grad_x = xp.zeros((batch, channels, H_padded, W_padded), dtype=g.dtype)
    for kh in range(kH):
        for kw in range(kW):
            grad_x[:, :, kh : kh + H_out * sH : sH, kw : kw + W_out * sW : sW] += (
                g_scaled
            )

    if padding > 0:
        grad_x = grad_x[:, :, padding:-padding, padding:-padding]
    return grad_x


def _batch_norm_backward(g, x, weight, bias, mean, var, eps=1e-5):
    xp = get_xp(x)
    x_norm = (x - mean) / xp.sqrt(var + eps)
    grad_x_norm = g * weight

    # Determine reduction axes: BatchNorm2d → (0,2,3); BatchNorm1d → (0,)
    if x.ndim == 4:
        norm_axes = (0, 2, 3)
        N = x.shape[0] * x.shape[2] * x.shape[3]
    else:
        norm_axes = (0,)
        N = x.shape[0]

    std = xp.sqrt(var + eps)
    grad_x = (
        (1.0 / N)
        * (1.0 / std)
        * (
            N * grad_x_norm
            - xp.sum(grad_x_norm, axis=norm_axes, keepdims=True)
            - x_norm * xp.sum(grad_x_norm * x_norm, axis=norm_axes, keepdims=True)
        )
    )
    grad_weight = unbroadcast(g * x_norm, weight.shape)
    grad_bias = unbroadcast(g, bias.shape)
    return grad_x, grad_weight, grad_bias


# ===========================================================================
# VJP rules — registered via @vjp_rule
# ===========================================================================

# ---------------------------------------------------------------------------
# Binary element-wise ops
# ---------------------------------------------------------------------------


@vjp_rule(func=np.add)
def _vjp_add(g, x, y):
    return (unbroadcast(g, x.shape), unbroadcast(g, y.shape))


@vjp_rule(func=np.subtract)
def _vjp_subtract(g, x, y):
    return (unbroadcast(g, x.shape), unbroadcast(-g, y.shape))


@vjp_rule(func=np.multiply)
def _vjp_multiply(g, x, y):
    return (unbroadcast(g * y, x.shape), unbroadcast(g * x, y.shape))


@vjp_rule(func=np.divide)
def _vjp_divide(g, x, y):
    return (
        unbroadcast(g / y, x.shape),
        unbroadcast(-g * x / (y**2 + safe_eps(y)), y.shape),
    )


@vjp_rule(func=np.power)
def _vjp_power(g, x, y):
    xp = get_xp(x)
    return (
        unbroadcast(g * y * xp.power(x, y - 1), x.shape),
        unbroadcast(g * xp.power(x, y) * xp.log(x + safe_eps(x)), y.shape),
    )


@vjp_rule(func=np.maximum)
def _vjp_maximum(g, x, y):
    return (unbroadcast(g * (x >= y), x.shape), unbroadcast(g * (x < y), y.shape))


@vjp_rule(func=np.minimum)
def _vjp_minimum(g, x, y):
    return (unbroadcast(g * (x <= y), x.shape), unbroadcast(g * (x > y), y.shape))


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------


@vjp_rule(func=np.negative)
def _vjp_negative(g, x):
    return (-g,)


@vjp_rule(func=np.square)
def _vjp_square(g, x):
    return (g * 2 * x,)


@vjp_rule(func=np.sqrt)
def _vjp_sqrt(g, x):
    return (g / (2 * get_xp(x).sqrt(x) + safe_eps(x)),)


@vjp_rule(func=np.exp)
def _vjp_exp(g, x):
    return (g * get_xp(x).exp(x),)


@vjp_rule(func=np.log)
def _vjp_log(g, x):
    return (g / (x + safe_eps(x)),)


@vjp_rule(func=np.log1p)
def _vjp_log1p(g, x):
    return (g / (x + 1.0 + safe_eps(x)),)


@vjp_rule(func=np.expm1)
def _vjp_expm1(g, x):
    return (g * get_xp(x).exp(x),)


@vjp_rule(func=np.abs)
def _vjp_abs(g, x):
    return (g * get_xp(x).sign(x),)


@vjp_rule(func=np.sign)
def _vjp_sign(g, x):
    return (get_xp(x).zeros_like(x),)


# ---------------------------------------------------------------------------
# Trigonometric & hyperbolic ops
# ---------------------------------------------------------------------------


@vjp_rule(func=np.sin)
def _vjp_sin(g, x):
    return (g * get_xp(x).cos(x),)


@vjp_rule(func=np.cos)
def _vjp_cos(g, x):
    return (g * -get_xp(x).sin(x),)


@vjp_rule(func=np.tan)
def _vjp_tan(g, x):
    return (g / (get_xp(x).cos(x) ** 2 + safe_eps(x)),)


@vjp_rule(func=np.sinh)
def _vjp_sinh(g, x):
    return (g * get_xp(x).cosh(x),)


@vjp_rule(func=np.cosh)
def _vjp_cosh(g, x):
    return (g * get_xp(x).sinh(x),)


@vjp_rule(func=np.tanh)
def _vjp_tanh(g, x):
    return (g * (1 - get_xp(x).tanh(x) ** 2),)


# ---------------------------------------------------------------------------
# Rounding ops  (zero gradient everywhere)
# ---------------------------------------------------------------------------


@vjp_rule(func=np.floor)
def _vjp_floor(g, x):
    return (get_xp(x).zeros_like(x),)


@vjp_rule(func=np.ceil)
def _vjp_ceil(g, x):
    return (get_xp(x).zeros_like(x),)


@vjp_rule(func=np.round)
def _vjp_round(g, x):
    return (get_xp(x).zeros_like(x),)


# ---------------------------------------------------------------------------
# Matrix / linear-algebra ops
# ---------------------------------------------------------------------------


@vjp_rule(func=np.matmul)
def _vjp_matmul(g, x, y):
    xp = get_xp(g)
    return (
        unbroadcast(
            xp.matmul(g, xp.swapaxes(y, -1, -2) if getattr(y, "ndim", 0) >= 2 else y),
            x.shape,
        ),
        unbroadcast(
            xp.matmul(xp.swapaxes(x, -1, -2) if getattr(x, "ndim", 0) >= 2 else x, g),
            y.shape,
        ),
    )


@vjp_rule(func=np.dot)
def _vjp_dot(g, x, y):
    xp = get_xp(g)
    return (xp.dot(g, y.T), xp.dot(x.T, g))


# ---------------------------------------------------------------------------
# Reduction ops
# ---------------------------------------------------------------------------


@vjp_rule(func=np.sum)
def _vjp_sum(g, x, axis=None, keepdims=False):
    return (_restore_reduced_dims(g, x.shape, axis, keepdims),)


@vjp_rule(func=np.mean)
def _vjp_mean(g, x, axis=None, keepdims=False):
    # x.shape[a] are pure Python ints — compute n entirely in Python,
    # avoiding a GPU alloc + sync just to multiply a few scalars.
    if axis is not None:
        axes = axis if isinstance(axis, (list, tuple)) else [axis]
        n = 1
        for a in axes:
            n *= x.shape[a]
    else:
        n = x.size
    return (_restore_reduced_dims(g, x.shape, axis, keepdims) / n,)


@vjp_rule(func=np.prod)
def _vjp_prod(g, x, axis=None, keepdims=False):
    xp = get_xp(x)
    return (
        _restore_reduced_dims(g, x.shape, axis, keepdims)
        * (xp.prod(x, axis=axis, keepdims=True) / (x + safe_eps(x))),
    )


@vjp_rule(func=np.max)
def _vjp_max(g, x, axis=None, keepdims=False):
    xp = get_xp(x)
    return (
        _restore_reduced_dims(g, x.shape, axis, keepdims)
        * (x == xp.max(x, axis=axis, keepdims=True)).astype(g.dtype),
    )


@vjp_rule(func=np.min)
def _vjp_min(g, x, axis=None, keepdims=False):
    xp = get_xp(x)
    return (
        _restore_reduced_dims(g, x.shape, axis, keepdims)
        * (x == xp.min(x, axis=axis, keepdims=True)).astype(g.dtype),
    )


# ---------------------------------------------------------------------------
# Shape / indexing ops
# ---------------------------------------------------------------------------


@vjp_rule(func=np.transpose)
def _vjp_transpose(g, x, axes=None):
    xp = get_xp(g)
    # np.argsort is used intentionally — axes is always a tiny Python list/tuple,
    # and cp.argsort(tuple) fails on some CuPy versions.  The result is passed
    # back as a plain list so both NumPy and CuPy accept it for transpose.
    return (
        xp.transpose(g, np.argsort(list(axes)).tolist()) if axes is not None else g.T,
    )


@vjp_rule(func=np.expand_dims)
def _vjp_expand_dims(g, x, axis):
    return (get_xp(g).squeeze(g, axis),)


@vjp_rule(func=np.squeeze)
def _vjp_squeeze(g, x, axis=None):
    xp = get_xp(g)
    return (xp.expand_dims(g, axis) if axis is not None else xp.reshape(g, x.shape),)


@vjp_rule(func=_reshape)
def _vjp_reshape(g, x, newshape):
    return (get_xp(g).reshape(g, x.shape),)


# Keep np.reshape as an alias so any code that looks up VJP_RULES[np.reshape]
# directly (e.g. legacy callers) still works.
VJP_RULES[np.reshape] = _vjp_reshape


# ---------------------------------------------------------------------------
# Neural-network activation ops
# ---------------------------------------------------------------------------


@vjp_rule(func=sigmoid)
def _vjp_sigmoid(g, x):
    s = sigmoid(x)
    return (g * s * (1.0 - s),)


@vjp_rule(func=relu)
def _vjp_relu(g, x):
    return (g * (x > 0).astype(x.dtype),)


@vjp_rule(func=leaky_relu)
def _vjp_leaky_relu(g, x):
    xp = get_xp(x)
    return (g * xp.where(x > 0, 1.0, 0.01),)


@vjp_rule(func=elu)
def _vjp_elu(g, x):
    xp = get_xp(x)
    return (g * xp.where(x > 0, 1.0, 1.0 * xp.exp(x)),)


@vjp_rule(func=softplus)
def _vjp_softplus(g, x):
    return (g * sigmoid(x),)


@vjp_rule(func=swish)
def _vjp_swish(g, x):
    s = sigmoid(x)
    sw = x * s
    return (g * (sw + s * (1.0 - sw)),)


@vjp_rule(func=gelu)
def _vjp_gelu(g, x):
    xp = get_xp(x)
    inner = _GELU_COEFF * (x + 0.044715 * x**3)
    t = xp.tanh(inner)  # computed once, reused three times
    dcdf = _GELU_COEFF * (1.0 + 3.0 * 0.044715 * x**2)
    return (g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * dcdf),)


@vjp_rule(func=softmax)
def _vjp_softmax(g, x):
    xp = get_xp(x)
    s = softmax(x)  # computed once, reused twice
    return (s * (g - xp.sum(g * s, axis=-1, keepdims=True)),)


# ---------------------------------------------------------------------------
# Convolution ops
# ---------------------------------------------------------------------------


@vjp_rule(func=conv2d)
def _vjp_conv2d(g, x, w, mode="valid"):
    return (
        conv2d(
            g,
            rot180(w),
            mode="full" if mode == "valid" else "same" if mode == "same" else "valid",
        ),
        conv2d(
            x
            if mode == "valid"
            else get_xp(x).pad(x, [(w.shape[0] // 2,) * 2, (w.shape[1] // 2,) * 2])
            if mode == "same"
            else get_xp(x).pad(x, [(w.shape[0] - 1,) * 2, (w.shape[1] - 1,) * 2]),
            g,
            mode="valid",
        ),
    )


@vjp_rule(func=_conv2d_forward_kernel)
def _vjp_conv2d_forward_kernel(
    g, x_unpad, W, pad_h=0, pad_w=0, stride_h=1, stride_w=1, H_out=1, W_out=1
):
    """
    VJP for _conv2d_forward_kernel(x_padded, W, stride_h, stride_w, H_out, W_out).

    Parameters
    ----------
    g        : (batch, out_ch, H_out, W_out)  upstream gradient
    x_unpad  : (batch, in_ch, H, W)           original unpadded input
    W        : (out_ch, in_ch, kH, kW)        filter weights
    pad_h, pad_w, stride_h, stride_w, H_out, W_out  : stored in op_kwargs

    Returns
    -------
    dx  : same shape as x_unpad
    dW  : same shape as W
    """
    xp = get_xp(g)
    batch = x_unpad.shape[0]
    out_ch, in_ch, kH, kW = W.shape

    if pad_h > 0 or pad_w > 0:
        x_padded = xp.pad(x_unpad, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    else:
        x_padded = x_unpad

    s = x_padded.strides
    col_shape = (batch, in_ch, kH, kW, H_out, W_out)
    col_strides = (s[0], s[1], s[2], s[3], stride_h * s[2], stride_w * s[3])
    cols = xp.lib.stride_tricks.as_strided(
        x_padded, shape=col_shape, strides=col_strides
    )
    cols_2d = xp.ascontiguousarray(cols).reshape(batch, in_ch * kH * kW, H_out * W_out)

    g_2d = g.reshape(batch, out_ch, H_out * W_out)
    W_2d = W.reshape(out_ch, in_ch * kH * kW)

    # Gradient w.r.t. W: dW[c, k] = sum_{b,n} g[b,c,n] * cols[b,k,n]
    dW_2d = xp.matmul(g_2d, cols_2d.transpose(0, 2, 1)).sum(axis=0)
    dW = dW_2d.reshape(W.shape)

    # Gradient w.r.t. cols via W^T @ g
    dcols_2d = xp.matmul(W_2d.T[None], g_2d)  # (1,K,C_out) @ (B,C_out,N) → (B,K,N)
    dcols = dcols_2d.reshape(batch, in_ch, kH, kW, H_out, W_out)

    dx_padded = xp.zeros_like(x_padded)
    for dh in range(kH):
        for dw in range(kW):
            dx_padded[
                :,
                :,
                dh : dh + H_out * stride_h : stride_h,
                dw : dw + W_out * stride_w : stride_w,
            ] += dcols[:, :, dh, dw, :, :]

    if pad_h > 0 and pad_w > 0:
        dx = dx_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
    elif pad_h > 0:
        dx = dx_padded[:, :, pad_h:-pad_h, :]
    elif pad_w > 0:
        dx = dx_padded[:, :, :, pad_w:-pad_w]
    else:
        dx = dx_padded

    return dx, dW


# ---------------------------------------------------------------------------
# Pooling ops
# ---------------------------------------------------------------------------


@vjp_rule(func=max_pool2d)
def _vjp_max_pool2d(g, x, kernel_size, stride=1, padding=0):
    return (_max_pool2d_backward(g, x, kernel_size, stride, padding),)


@vjp_rule(func=avg_pool2d)
def _vjp_avg_pool2d(g, x, kernel_size, stride=1, padding=0):
    return (_avg_pool2d_backward(g, x, kernel_size, stride, padding),)


# ---------------------------------------------------------------------------
# Regularization & normalization ops
# ---------------------------------------------------------------------------


@vjp_rule(func=dropout)
def _vjp_dropout(g, x, p=0.5, training=True, mask=None):
    # Use the recorded binary mask; fall back gracefully if absent.
    if training and mask is not None:
        return (g * mask,)
    if training:
        return (g / (1 - p),)
    return (g,)


@vjp_rule(func=batch_norm)
def _vjp_batch_norm(g, x, weight, bias, mean, var, eps=1e-5):
    return _batch_norm_backward(g, x, weight, bias, mean, var, eps)


# ---------------------------------------------------------------------------
# Array manipulation ops (variadic / numpy-native)
# ---------------------------------------------------------------------------


@vjp_rule(func=_concatenate)
def _vjp_concatenate(g, *arrays, axis=0):
    """Reverse of concatenate: split upstream gradient at input boundaries."""
    xp = get_xp(g)
    split_indices = list(np.cumsum([a.shape[axis] for a in arrays[:-1]], dtype=int))
    parts = xp.split(g, split_indices, axis=axis)
    return tuple(parts)


@vjp_rule(func=_stack)
def _vjp_stack(g, *arrays, axis=0):
    """Reverse of stack: select each slice along the stacked axis."""
    return tuple(g[(slice(None),) * axis + (i,)] for i in range(len(arrays)))


@vjp_rule(func=np.clip)
def _vjp_clip(g, x, a_min=None, a_max=None):
    """Gradient flows only where the input is inside (a_min, a_max)."""
    xp = get_xp(x)
    mask = xp.ones(x.shape, dtype=bool)
    if a_min is not None:
        mask = mask & (x >= a_min)
    if a_max is not None:
        mask = mask & (x <= a_max)
    return (g * mask.astype(g.dtype),)


@vjp_rule(func=np.cumsum)
def _vjp_cumsum(g, x, axis=None):
    """VJP of cumsum: reverse cumulative sum along the given axis."""
    xp = get_xp(g)
    if axis is None:
        g_flat = g.reshape(-1)
        return (xp.flip(xp.cumsum(xp.flip(g_flat, 0), 0), 0).reshape(x.shape),)
    return (xp.flip(xp.cumsum(xp.flip(g, axis), axis), axis),)


@vjp_rule(func=np.flip)
def _vjp_flip(g, x, axis=None):
    """VJP of flip: flip is its own inverse."""
    xp = get_xp(g)
    return (xp.flip(g, axis),)


@vjp_rule(func=np.roll)
def _vjp_roll(g, x, shift, axis=None):
    """VJP of roll: un-roll by shifting in the opposite direction."""
    xp = get_xp(g)
    return (xp.roll(g, -shift, axis),)


@vjp_rule(func=np.tile)
def _vjp_tile(g, x, reps):
    """VJP of tile: fold tiled copies back into x's shape and sum."""
    xp = get_xp(g)
    # reps may arrive as a 0-d ndarray (from scalar wrapping in vpj) or int.
    if hasattr(reps, "ndim") and reps.ndim == 0:
        reps = (int(reps),)
    elif isinstance(reps, int):
        reps = (reps,)
    reps = tuple(int(r) for r in reps)
    n = max(len(reps), x.ndim)
    reps_padded = (1,) * (n - len(reps)) + reps
    x_shape = (1,) * (n - x.ndim) + x.shape
    # Reshape g to interleave (rep, size) pairs, then sum the rep-axes.
    interleaved = [s for pair in zip(reps_padded, x_shape) for s in pair]
    sum_axes = tuple(range(0, 2 * n, 2))
    return (
        xp.ascontiguousarray(g)
        .reshape(interleaved)
        .sum(axis=sum_axes)
        .reshape(x.shape),
    )


@vjp_rule(func=np.repeat)
def _vjp_repeat(g, x, repeats, axis=None):
    """VJP of repeat: accumulate gradient from all repeated copies."""
    xp = get_xp(g)
    # repeats may arrive as a 0-d ndarray (scalar-wrapped by vpj).
    scalar_reps = isinstance(repeats, int) or (
        hasattr(repeats, "ndim") and repeats.ndim == 0
    )
    if scalar_reps:
        repeats = int(repeats)
    if axis is None:
        g_flat = g.reshape(-1)
        if scalar_reps:
            return (g_flat.reshape(-1, repeats).sum(axis=1).reshape(x.shape),)
        # Variable repeats along the flattened view
        splits = list(np.cumsum(np.asarray(repeats)[:-1]).astype(int))
        parts = xp.split(g_flat, splits)
        # xp.stack keeps everything on the same device — no float() CPU sync.
        return (xp.stack([p.sum() for p in parts]).astype(g.dtype).reshape(x.shape),)
    if scalar_reps:
        g_shape = list(g.shape)
        new_shape = g_shape[:axis] + [x.shape[axis], repeats] + g_shape[axis + 1 :]
        return (xp.ascontiguousarray(g).reshape(new_shape).sum(axis=axis + 1),)
    # Variable repeats along a specific axis
    splits = list(np.cumsum(np.asarray(repeats)[:-1]).astype(int))
    parts = xp.split(g, splits, axis=axis)
    return (xp.stack([p.sum(axis=axis) for p in parts], axis=axis),)
