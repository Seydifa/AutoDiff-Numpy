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
import numpy as _np
from .backend import backend, safe_eps, get_dtype
from .tensor import Tensor
from .session import session


# Precomputed constants — avoids backend calls on every forward pass.
_GELU_COEFF: float = 0.7978845608028654  # sqrt(2/pi)
_LOG2: float = 0.6931471805599453  # log(2.0)

# Deprecated — kept for backward compatibility; use safe_eps() instead.
EPSILON: float = 1e-8

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

        @vjp_rule(func=backend.sin)
        def _vjp_sin(g, x):
            return (g * backend.cos(x),)

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
    """Re-insert squeezed axes into *g* so it broadcasts to *x_shape*.

    Uses ``backend.broadcast_to`` (zero-copy view) instead of ``backend.ones`` allocation.
    Handles axis=None, int, list, and tuple.  Works on NumPy < 2.0 (no tuple
    axis in expand_dims).
    """
    if keepdims or axis is None:
        return backend.broadcast_to(g, x_shape)
    if isinstance(axis, (list, tuple)):
        result = g
        for ax in sorted(int(a) % len(x_shape) for a in axis):
            result = backend.expand_dims(result, ax)
    else:
        result = backend.expand_dims(g, int(axis) % len(x_shape))
    return backend.broadcast_to(result, x_shape)


# ---------------------------------------------------------------------------
# Activation / helper functions (also serve as forward-pass callables)
# ---------------------------------------------------------------------------


def sigmoid(x):
    x_clipped = backend.clip(x, -500, 500)
    return 1.0 / (1.0 + backend.exp(-x_clipped))


def relu(x):
    return backend.maximum(0.0, x)


def leaky_relu(x, alpha=0.01):
    return backend.where(x > 0, x, alpha * x)


def elu(x, alpha=1.0):
    return backend.where(x > 0, x, alpha * (backend.exp(x) - 1.0))


def softplus(x):
    # logaddexp(0, x) = log(1 + exp(x)) computed in a numerically stable way.
    return backend.logaddexp(0.0, x)


def swish(x):
    return x * sigmoid(x)


def gelu(x):
    # Use module-level constant (no sqrt per call) and ** instead of backend.power.
    return 0.5 * x * (1.0 + backend.tanh(_GELU_COEFF * (x + 0.044715 * x**3)))


def softmax(x, axis=-1):
    x_max = backend.max(x, axis=axis, keepdims=True)
    e_x = backend.exp(x - x_max)
    return e_x / backend.sum(e_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Pooling operations (vectorized)
# ---------------------------------------------------------------------------


def max_pool2d(x, kernel_size, stride=1, padding=0):

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    if padding > 0:
        x = backend.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

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

    x_windowed = backend.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return backend.max(x_windowed, axis=(4, 5))


def avg_pool2d(x, kernel_size, stride=1, padding=0):

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    if padding > 0:
        x = backend.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

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

    x_windowed = backend.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return backend.mean(x_windowed, axis=(4, 5))


def dropout(x, p=0.5, training=True):
    """Dropout forward — the binary mask is generated in ``_DropoutOps._raw_call``
    and stored in ``op_kwargs["mask"]`` so ``Tensor.backward()`` can retrieve it.
    """
    if not training or p == 0:
        return x
    mask = (backend.random.uniform(size=x.shape) >= p).astype(x.dtype)
    return x * mask / (1 - p)


def batch_norm(x, weight, bias, mean, var, eps=1e-5):
    x_norm = (x - mean) / backend.sqrt(var + eps)
    return weight * x_norm + bias


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------


def conv2d(x, w, mode="valid"):
    """2D convolution — stays on whichever device x lives on.

    Delegates entirely to ``backend.scipy.signal.convolve2d``, which resolves
    to ``cupyx.scipy.signal`` on GPU or ``scipy.signal`` on CPU — no manual
    branching or inline imports needed.
    """
    return backend.scipy.signal.convolve2d(x, w, mode=mode)


def rot180(w):
    return backend.rot90(w, 2)


def conv2d_full(g, w):
    return conv2d(g, w, mode="full")


def conv2d_nd(x, W, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
    if pad_h > 0 or pad_w > 0:
        x_padded = backend.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    else:
        x_padded = x

    kH, kW = W.shape[2], W.shape[3]
    H_out = (x_padded.shape[2] - kH) // stride_h + 1
    W_out = (x_padded.shape[3] - kW) // stride_w + 1

    return _conv2d_forward_kernel(x_padded, W, stride_h, stride_w, H_out, W_out)


# ---------------------------------------------------------------------------
# Im2col convolution kernel  (forward callable — also the VJP dict key)
# ---------------------------------------------------------------------------


def _conv2d_forward_kernel(x_padded, W, stride_h, stride_w, H_out, W_out):
    """Im2col + batched matmul convolution — O(B·Cout·Cin·kH·kW·Hout·Wout).

    Replaces the old 6-level Numba loop with a fully vectorized path that
    works transparently on both NumPy (CPU) and CuPy (GPU) arrays.
    """
    batch, in_ch, _, _ = x_padded.shape
    out_ch, _, kH, kW = W.shape

    # Build a strided patch view: (batch, in_ch, kH, kW, H_out, W_out)
    s = x_padded.strides
    col_shape = (batch, in_ch, kH, kW, H_out, W_out)
    col_strides = (s[0], s[1], s[2], s[3], stride_h * s[2], stride_w * s[3])
    cols = backend.lib.stride_tricks.as_strided(
        x_padded, shape=col_shape, strides=col_strides
    )
    cols_2d = backend.ascontiguousarray(cols).reshape(
        batch, in_ch * kH * kW, H_out * W_out
    )
    W_2d = W.reshape(out_ch, in_ch * kH * kW)
    return backend.matmul(W_2d, cols_2d).reshape(batch, out_ch, H_out, W_out)


# ---------------------------------------------------------------------------
# NumPy-version-safe reshape wrapper
# ---------------------------------------------------------------------------
# NumPy 2.0 renamed the ``newshape`` keyword to ``shape``.  Passing the new
# shape *positionally* works on every version, so we wrap it here.


def _reshape(a, newshape):
    """Backend-agnostic reshape — stays on whichever device *a* lives on."""
    return backend.reshape(a, newshape)


# ---------------------------------------------------------------------------
# Variadic forward wrappers — individual Tensor tracking per input
# ---------------------------------------------------------------------------


def _concatenate(*arrays, axis=0):
    """Variadic wrapper: ``_concatenate(a, b, c, axis=0)`` → backend concatenate."""
    return backend.concatenate(arrays, axis=axis)


def _stack(*arrays, axis=0):
    """Variadic wrapper: ``_stack(a, b, c, axis=0)`` → backend stack."""
    return backend.stack(arrays, axis=axis)


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
    """Fully vectorized max-pool backward — zero Python loops over spatial dims.

    Strategy
    --------
    1. Build the strided window view  (B, C, H_out, W_out, kH, kW)
    2. Compute the tie-broken argmax flat index inside each window.
    3. Convert to absolute (h, w) coordinates in the padded input.
    4. Scatter-add the upstream gradient into grad_x in *one* vectorized call
       using advanced indexing — no Python loop over kH, kW, or spatial dims.

    Works identically on NumPy (CPU) and CuPy (GPU); no data transfer.
    """
    kernel_size = _norm_pair(kernel_size)
    stride = _norm_pair(stride)
    padding = _norm_int(padding)
    kH, kW = kernel_size
    sH, sW = stride

    if padding > 0:
        x = backend.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    batch, channels, H_pad, W_pad = x.shape
    H_out = (H_pad - kH) // sH + 1
    W_out = (W_pad - kW) // sW + 1

    # ---- strided window view: (B, C, H_out, W_out, kH, kW) ----------------
    s = x.strides
    win_shape = (batch, channels, H_out, W_out, kH, kW)
    win_strides = (s[0], s[1], sH * s[2], sW * s[3], s[2], s[3])
    x_win = backend.lib.stride_tricks.as_strided(
        x, shape=win_shape, strides=win_strides
    )

    # ---- argmax (flat within kH*kW window) ---------------------------------
    # reshape to (B, C, H_out, W_out, kH*kW) for argmax
    x_flat = x_win.reshape(batch, channels, H_out, W_out, kH * kW)
    flat_idx = backend.argmax(x_flat, axis=4)  # (B, C, H_out, W_out)

    # ---- convert flat_idx → (kh_idx, kw_idx) -------------------------------
    kh_idx = flat_idx // kW  # (B, C, H_out, W_out)
    kw_idx = flat_idx % kW

    # ---- absolute (h, w) position in padded input --------------------------
    # h_out_idx, w_out_idx: broadcast shapes (1, 1, H_out, 1) etc.
    h_out_idx = backend.arange(H_out, dtype=backend.int64).reshape(1, 1, H_out, 1)
    w_out_idx = backend.arange(W_out, dtype=backend.int64).reshape(1, 1, 1, W_out)

    abs_h = h_out_idx * sH + kh_idx  # (B, C, H_out, W_out)
    abs_w = w_out_idx * sW + kw_idx

    # ---- batch & channel index grids (for advanced indexing) ---------------
    b_idx = backend.arange(batch, dtype=backend.int64).reshape(batch, 1, 1, 1)
    c_idx = backend.arange(channels, dtype=backend.int64).reshape(1, channels, 1, 1)

    # ---- scatter-add in one vectorized call --------------------------------
    grad_x = backend.zeros_like(x)
    # backend.add.at works on both numpy and cupy
    backend.add.at(grad_x, (b_idx, c_idx, abs_h, abs_w), g)

    if padding > 0:
        grad_x = grad_x[:, :, padding:-padding, padding:-padding]
    return grad_x


def _avg_pool2d_backward(g, x, kernel_size, stride=1, padding=0):
    """Fully vectorized avg-pool backward — zero Python loops.

    Strategy
    --------
    Build (kH, kW) offset grids once with meshgrid, then scatter-add the
    scaled upstream gradient for *all* kernel offsets simultaneously using
    advanced indexing — no Python loops at all.

    Works identically on NumPy (CPU) and CuPy (GPU); no data transfer.
    """
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

    g_scaled = g / (kH * kW)  # (B, C, H_out, W_out)

    # Build all kernel-offset positions at once: kh_grid/kw_grid shape (kH, kW)
    kh_range = backend.arange(kH, dtype=backend.int64)
    kw_range = backend.arange(kW, dtype=backend.int64)
    kh_grid, kw_grid = backend.meshgrid(kh_range, kw_range, indexing="ij")
    # Flatten to (kH*kW,)
    kh_flat = kh_grid.ravel()  # (K,) where K=kH*kW
    kw_flat = kw_grid.ravel()

    # Output position grids: h_out_grid/w_out_grid shape (H_out,) and (W_out,)
    h_out_range = backend.arange(H_out, dtype=backend.int64)
    w_out_range = backend.arange(W_out, dtype=backend.int64)

    # Absolute input positions for every (output pos, kernel offset) pair:
    # abs_h shape: (K, H_out, 1) after broadcasting
    # abs_w shape: (K, 1, W_out)
    abs_h = kh_flat[:, None, None] + h_out_range[None, :, None] * sH  # (K, H_out, 1)
    abs_w = kw_flat[:, None, None] + w_out_range[None, None, :] * sW  # (K, 1, W_out)

    # Index grids for batch and channel: (batch,1,1,1,1) etc.
    b_idx = backend.arange(batch, dtype=backend.int64)[:, None, None, None, None]
    c_idx = backend.arange(channels, dtype=backend.int64)[None, :, None, None, None]
    # abs_h/w: add K dim → (1,1,K,H_out,1) and (1,1,K,1,W_out)
    h_idx = abs_h[None, None, :, :, :]  # (1, 1, K, H_out, 1)
    w_idx = abs_w[None, None, :, :, :]  # (1, 1, K, 1, W_out)
    # g_scaled needs K dim: (B, C, 1, H_out, W_out)
    g_exp = g_scaled[:, :, None, :, :]  # broadcast over K

    grad_x = backend.zeros((batch, channels, H_padded, W_padded), dtype=g.dtype)
    backend.add.at(grad_x, (b_idx, c_idx, h_idx, w_idx), g_exp)

    if padding > 0:
        grad_x = grad_x[:, :, padding:-padding, padding:-padding]
    return grad_x


def _batch_norm_backward(g, x, weight, bias, mean, var, eps=1e-5):
    x_norm = (x - mean) / backend.sqrt(var + eps)
    grad_x_norm = g * weight

    # Determine reduction axes: BatchNorm2d → (0,2,3); BatchNorm1d → (0,)
    if x.ndim == 4:
        norm_axes = (0, 2, 3)
        N = x.shape[0] * x.shape[2] * x.shape[3]
    else:
        norm_axes = (0,)
        N = x.shape[0]

    std = backend.sqrt(var + eps)
    grad_x = (
        (1.0 / N)
        * (1.0 / std)
        * (
            N * grad_x_norm
            - backend.sum(grad_x_norm, axis=norm_axes, keepdims=True)
            - x_norm * backend.sum(grad_x_norm * x_norm, axis=norm_axes, keepdims=True)
        )
    )
    grad_weight = unbroadcast(g * x_norm, weight.shape)
    grad_bias = unbroadcast(g, bias.shape)
    return grad_x, grad_weight, grad_bias


# ---------------------------------------------------------------------------
# Pure-Python helpers (zero device overhead)
# ---------------------------------------------------------------------------


def _py_cumsum(seq):
    """Return a plain list of prefix-sums (no numpy/cupy allocation)."""
    out, total = [], 0
    for v in seq:
        total += v
        out.append(total)
    return out


# ===========================================================================
# VJP rules — registered via @vjp_rule
# ===========================================================================

# ---------------------------------------------------------------------------
# Binary element-wise ops
# ---------------------------------------------------------------------------


@vjp_rule(func=backend.add)
def _vjp_add(g, x, y):
    return (unbroadcast(g, x.shape), unbroadcast(g, y.shape))


@vjp_rule(func=backend.subtract)
def _vjp_subtract(g, x, y):
    return (unbroadcast(g, x.shape), unbroadcast(-g, y.shape))


@vjp_rule(func=backend.multiply)
def _vjp_multiply(g, x, y):
    return (unbroadcast(g * y, x.shape), unbroadcast(g * x, y.shape))


@vjp_rule(func=backend.divide)
def _vjp_divide(g, x, y):
    return (
        unbroadcast(g / y, x.shape),
        unbroadcast(-g * x / (y**2 + safe_eps(y)), y.shape),
    )


@vjp_rule(func=backend.power)
def _vjp_power(g, x, y):
    return (
        unbroadcast(g * y * backend.power(x, y - 1), x.shape),
        unbroadcast(g * backend.power(x, y) * backend.log(x + safe_eps(x)), y.shape),
    )


@vjp_rule(func=backend.maximum)
def _vjp_maximum(g, x, y):
    return (unbroadcast(g * (x >= y), x.shape), unbroadcast(g * (x < y), y.shape))


@vjp_rule(func=backend.minimum)
def _vjp_minimum(g, x, y):
    return (unbroadcast(g * (x <= y), x.shape), unbroadcast(g * (x > y), y.shape))


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------


@vjp_rule(func=backend.negative)
def _vjp_negative(g, x):
    return (-g,)


@vjp_rule(func=backend.square)
def _vjp_square(g, x):
    return (g * 2 * x,)


@vjp_rule(func=backend.sqrt)
def _vjp_sqrt(g, x):
    return (g / (2 * backend.sqrt(x) + safe_eps(x)),)


@vjp_rule(func=backend.exp)
def _vjp_exp(g, x):
    return (g * backend.exp(x),)


@vjp_rule(func=backend.log)
def _vjp_log(g, x):
    return (g / (x + safe_eps(x)),)


@vjp_rule(func=backend.log1p)
def _vjp_log1p(g, x):
    return (g / (x + 1.0 + safe_eps(x)),)


@vjp_rule(func=backend.expm1)
def _vjp_expm1(g, x):
    return (g * backend.exp(x),)


@vjp_rule(func=backend.abs)
def _vjp_abs(g, x):
    return (g * backend.sign(x),)


@vjp_rule(func=backend.sign)
def _vjp_sign(g, x):
    return (backend.zeros_like(x),)


# ---------------------------------------------------------------------------
# Trigonometric & hyperbolic ops
# ---------------------------------------------------------------------------


@vjp_rule(func=backend.sin)
def _vjp_sin(g, x):
    return (g * backend.cos(x),)


@vjp_rule(func=backend.cos)
def _vjp_cos(g, x):
    return (g * -backend.sin(x),)


@vjp_rule(func=backend.tan)
def _vjp_tan(g, x):
    return (g / (backend.cos(x) ** 2 + safe_eps(x)),)


@vjp_rule(func=backend.sinh)
def _vjp_sinh(g, x):
    return (g * backend.cosh(x),)


@vjp_rule(func=backend.cosh)
def _vjp_cosh(g, x):
    return (g * backend.sinh(x),)


@vjp_rule(func=backend.tanh)
def _vjp_tanh(g, x):
    return (g * (1 - backend.tanh(x) ** 2),)


# ---------------------------------------------------------------------------
# Special / Custom Ops
# ---------------------------------------------------------------------------


@vjp_rule(func=backend.where)
def _vjp_where(g, condition, x, y):
    zeros = backend.zeros_like(g)
    return (
        None,
        backend.where(condition, g, zeros),
        backend.where(condition, zeros, g),
    )


# ---------------------------------------------------------------------------
# Rounding ops  (zero gradient everywhere)
# ---------------------------------------------------------------------------


@vjp_rule(func=backend.floor)
def _vjp_floor(g, x):
    return (backend.zeros_like(x),)


@vjp_rule(func=backend.ceil)
def _vjp_ceil(g, x):
    return (backend.zeros_like(x),)


@vjp_rule(func=backend.round)
def _vjp_round(g, x):
    return (backend.zeros_like(x),)


# ---------------------------------------------------------------------------
# Matrix / linear-algebra ops
# ---------------------------------------------------------------------------


@vjp_rule(func=backend.matmul)
def _vjp_matmul(g, x, y):
    return (
        unbroadcast(
            backend.matmul(
                g, backend.swapaxes(y, -1, -2) if getattr(y, "ndim", 0) >= 2 else y
            ),
            x.shape,
        ),
        unbroadcast(
            backend.matmul(
                backend.swapaxes(x, -1, -2) if getattr(x, "ndim", 0) >= 2 else x, g
            ),
            y.shape,
        ),
    )


@vjp_rule(func=backend.dot)
def _vjp_dot(g, x, y):
    return (backend.dot(g, y.T), backend.dot(x.T, g))


# ---------------------------------------------------------------------------
# Reduction ops
# ---------------------------------------------------------------------------


@vjp_rule(func=backend.sum)
def _vjp_sum(g, x, axis=None, keepdims=False):
    return (_restore_reduced_dims(g, x.shape, axis, keepdims),)


@vjp_rule(func=backend.mean)
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


@vjp_rule(func=backend.prod)
def _vjp_prod(g, x, axis=None, keepdims=False):
    return (
        _restore_reduced_dims(g, x.shape, axis, keepdims)
        * (backend.prod(x, axis=axis, keepdims=True) / (x + safe_eps(x))),
    )


@vjp_rule(func=backend.max)
def _vjp_max(g, x, axis=None, keepdims=False):
    return (
        _restore_reduced_dims(g, x.shape, axis, keepdims)
        * (x == backend.max(x, axis=axis, keepdims=True)).astype(g.dtype),
    )


@vjp_rule(func=backend.min)
def _vjp_min(g, x, axis=None, keepdims=False):
    return (
        _restore_reduced_dims(g, x.shape, axis, keepdims)
        * (x == backend.min(x, axis=axis, keepdims=True)).astype(g.dtype),
    )


# ---------------------------------------------------------------------------
# Shape / indexing ops
# ---------------------------------------------------------------------------


@vjp_rule(func=backend.transpose)
def _vjp_transpose(g, x, axes=None):
    # backend.argsort is used intentionally — axes is always a tiny Python list/tuple,
    # and cp.argsort(tuple) fails on some CuPy versions.  The result is passed
    # back as a plain list so both NumPy and CuPy accept it for transpose.
    return (
        backend.transpose(g, _np.argsort(list(axes)).tolist())
        if axes is not None
        else g.T,
    )


@vjp_rule(func=backend.expand_dims)
def _vjp_expand_dims(g, x, axis):
    return (backend.squeeze(g, axis),)


@vjp_rule(func=backend.squeeze)
def _vjp_squeeze(g, x, axis=None):
    return (
        backend.expand_dims(g, axis)
        if axis is not None
        else backend.reshape(g, x.shape),
    )


@vjp_rule(func=_reshape)
def _vjp_reshape(g, x, newshape):
    return (backend.reshape(g, x.shape),)


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
def _vjp_leaky_relu(g, x, alpha=0.01):
    return (g * backend.where(x > 0, 1.0, alpha),)


@vjp_rule(func=elu)
def _vjp_elu(g, x, alpha=1.0):
    return (g * backend.where(x > 0, 1.0, alpha * backend.exp(x)),)


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
    inner = _GELU_COEFF * (x + 0.044715 * x**3)
    t = backend.tanh(inner)  # computed once, reused three times
    dcdf = _GELU_COEFF * (1.0 + 3.0 * 0.044715 * x**2)
    return (g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * dcdf),)


@vjp_rule(func=softmax)
def _vjp_softmax(g, x):
    s = softmax(x)  # computed once, reused twice
    return (s * (g - backend.sum(g * s, axis=-1, keepdims=True)),)


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
            else backend.pad(x, [(w.shape[0] // 2,) * 2, (w.shape[1] // 2,) * 2])
            if mode == "same"
            else backend.pad(x, [(w.shape[0] - 1,) * 2, (w.shape[1] - 1,) * 2]),
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
    batch = x_unpad.shape[0]
    out_ch, in_ch, kH, kW = W.shape

    if pad_h > 0 or pad_w > 0:
        x_padded = backend.pad(
            x_unpad, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w))
        )
    else:
        x_padded = x_unpad

    s = x_padded.strides
    col_shape = (batch, in_ch, kH, kW, H_out, W_out)
    col_strides = (s[0], s[1], s[2], s[3], stride_h * s[2], stride_w * s[3])
    cols = backend.lib.stride_tricks.as_strided(
        x_padded, shape=col_shape, strides=col_strides
    )
    cols_2d = backend.ascontiguousarray(cols).reshape(
        batch, in_ch * kH * kW, H_out * W_out
    )

    g_2d = g.reshape(batch, out_ch, H_out * W_out)
    W_2d = W.reshape(out_ch, in_ch * kH * kW)

    # Gradient w.r.t. W: dW[c, k] = sum_{b,n} g[b,c,n] * cols[b,k,n]
    dW_2d = backend.matmul(g_2d, cols_2d.transpose(0, 2, 1)).sum(axis=0)
    dW = dW_2d.reshape(W.shape)

    # Gradient w.r.t. cols via W^T @ g
    dcols_2d = backend.matmul(W_2d.T[None], g_2d)  # (1,K,C_out) @ (B,C_out,N) → (B,K,N)
    dcols = dcols_2d.reshape(batch, in_ch, kH, kW, H_out, W_out)

    dx_padded = backend.zeros_like(x_padded)

    # Vectorized col2im: scatter dcols into dx_padded in a single add.at call.
    # dcols shape: (batch, in_ch, kH, kW, H_out, W_out)
    # Each (b, c, dh, dw, ho, wo) maps to dx_padded[b, c, dh+ho*sH, dw+wo*sW].
    # Build broadcasted index arrays shaped (B,1,kH,1,H_out,1) etc. to match dcols.
    kh_range = backend.arange(kH, dtype=backend.int64)  # (kH,)
    kw_range = backend.arange(kW, dtype=backend.int64)  # (kW,)
    ho_range = backend.arange(H_out, dtype=backend.int64)  # (H_out,)
    wo_range = backend.arange(W_out, dtype=backend.int64)  # (W_out,)

    # abs positions: kh + ho*stride_h, kw + wo*stride_w
    abs_h = kh_range[:, None] + ho_range[None, :] * stride_h  # (kH, H_out)
    abs_w = kw_range[:, None] + wo_range[None, :] * stride_w  # (kW, W_out)

    b_full = backend.arange(batch, dtype=backend.int64).reshape(batch, 1, 1, 1, 1, 1)
    c_full = backend.arange(in_ch, dtype=backend.int64).reshape(1, in_ch, 1, 1, 1, 1)
    h_full = abs_h.reshape(1, 1, kH, 1, H_out, 1)
    w_full = abs_w.reshape(1, 1, 1, kW, 1, W_out)

    backend.add.at(dx_padded, (b_full, c_full, h_full, w_full), dcols)

    if pad_h > 0 and pad_w > 0:
        dx = dx_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
    elif pad_h > 0:
        dx = dx_padded[:, :, pad_h:-pad_h, :]
    elif pad_w > 0:
        dx = dx_padded[:, :, :, pad_w:-pad_w]
    else:
        dx = dx_padded

    return dx, dW


@vjp_rule(func=conv2d_nd)
def _vjp_conv2d_nd(g, x_unpad, W, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
    kH, kW = W.shape[2], W.shape[3]
    H_out = (x_unpad.shape[2] + 2 * pad_h - kH) // stride_h + 1
    W_out = (x_unpad.shape[3] + 2 * pad_w - kW) // stride_w + 1
    return _vjp_conv2d_forward_kernel(
        g, x_unpad, W, pad_h, pad_w, stride_h, stride_w, H_out, W_out
    )


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
    split_indices = list(
        backend.cumsum([a.shape[axis] for a in arrays[:-1]], dtype=int)
    )
    parts = backend.split(g, split_indices, axis=axis)
    return tuple(parts)


@vjp_rule(func=_stack)
def _vjp_stack(g, *arrays, axis=0):
    """Reverse of stack: select each slice along the stacked axis."""
    return tuple(g[(slice(None),) * axis + (i,)] for i in range(len(arrays)))


@vjp_rule(func=backend.clip)
def _vjp_clip(g, x, a_min=None, a_max=None):
    """Gradient flows only where the input is inside (a_min, a_max)."""
    mask = backend.ones(x.shape, dtype=bool)
    if a_min is not None:
        mask = mask & (x >= a_min)
    if a_max is not None:
        mask = mask & (x <= a_max)
    return (g * mask.astype(g.dtype),)


@vjp_rule(func=backend.cumsum)
def _vjp_cumsum(g, x, axis=None):
    """VJP of cumsum: reverse cumulative sum along the given axis."""
    if axis is None:
        g_flat = g.reshape(-1)
        return (
            backend.flip(backend.cumsum(backend.flip(g_flat, 0), 0), 0).reshape(
                x.shape
            ),
        )
    return (backend.flip(backend.cumsum(backend.flip(g, axis), axis), axis),)


@vjp_rule(func=backend.flip)
def _vjp_flip(g, x, axis=None):
    """VJP of flip: flip is its own inverse."""
    return (backend.flip(g, axis),)


@vjp_rule(func=backend.roll)
def _vjp_roll(g, x, shift, axis=None):
    """VJP of roll: un-roll by shifting in the opposite direction."""
    return (backend.roll(g, -shift, axis),)


@vjp_rule(func=backend.tile)
def _vjp_tile(g, x, reps):
    """VJP of tile: fold tiled copies back into x's shape and sum."""
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
        backend.ascontiguousarray(g)
        .reshape(interleaved)
        .sum(axis=sum_axes)
        .reshape(x.shape),
    )


@vjp_rule(func=backend.repeat)
def _vjp_repeat(g, x, repeats, axis=None):
    """VJP of repeat: accumulate gradient from all repeated copies.

    ``repeats`` is always a Python int or a plain Python list/tuple of ints
    by the time it reaches here (normalised by the calling Ops layer).
    We therefore use pure-Python cumsum to build split indices — no numpy/cupy
    allocation, no device sync.
    """
    # Normalise repeats to a plain Python int or list[int] — no device arrays.
    if hasattr(repeats, "ndim") and repeats.ndim == 0:
        repeats = int(repeats)
    scalar_reps = isinstance(repeats, int)
    if not scalar_reps:
        # Bring to CPU ints once, cheaply (it's a tiny 1-D metadata array).
        repeats = [
            int(r)
            for r in (repeats.tolist() if hasattr(repeats, "tolist") else repeats)
        ]

    if axis is None:
        g_flat = g.reshape(-1)
        if scalar_reps:
            return (g_flat.reshape(-1, repeats).sum(axis=1).reshape(x.shape),)
        # Pure-Python prefix-sum for split points — zero device overhead.
        splits = _py_cumsum(repeats[:-1])
        parts = backend.split(g_flat, splits)
        return (
            backend.stack([p.sum() for p in parts]).astype(g.dtype).reshape(x.shape),
        )
    if scalar_reps:
        g_shape = list(g.shape)
        new_shape = g_shape[:axis] + [x.shape[axis], repeats] + g_shape[axis + 1 :]
        return (backend.ascontiguousarray(g).reshape(new_shape).sum(axis=axis + 1),)
    splits = _py_cumsum(repeats[:-1])
    parts = backend.split(g, splits, axis=axis)
    return (backend.stack([p.sum(axis=axis) for p in parts], axis=axis),)


# ===========================================================================
# Loss functions — first-class differentiable ops
# ===========================================================================
#
# Design contract
# ---------------
# * Every forward function returns a *scalar* (mean-reduced) loss unless noted.
# * Every VJP is derived analytically from the closed-form definition —
#   no approximations, no finite differences, no composition through ops.
# * All ops are backend-agnostic: backend.* dispatches to CuPy on GPU.
# * Numerically stable implementations use log-sum-exp / fused log-sigmoid
#   tricks wherever cancellation would occur in the naive formula.
# * Shape contract: predictions first, targets second (matching PyTorch/Keras).
#
# Reduction convention
# --------------------
# All losses default to mean reduction over the batch (and spatial) dims.
# The upstream gradient `g` arriving from the graph is therefore a scalar;
# the per-element factor 1/N absorbs the mean.

# ---------------------------------------------------------------------------
# 1. Mean Squared Error  —  MSE
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def mse_loss(y_pred, y_true):
    """Mean squared error: mean((ŷ - y)²)."""
    diff = y_pred - y_true
    return backend.mean(diff * diff)


@vjp_rule(func=mse_loss)
def _vjp_mse_loss(g, y_pred, y_true):
    N = y_pred.size
    diff = y_pred - y_true
    # ∂L/∂ŷ = 2·diff/N,  ∂L/∂y = -2·diff/N
    grad = g * (2.0 / N) * diff
    return (grad, -grad)


# ---------------------------------------------------------------------------
# 2. Mean Absolute Error  —  MAE / L1
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def mae_loss(y_pred, y_true):
    """Mean absolute error: mean(|ŷ - y|)."""
    return backend.mean(backend.abs(y_pred - y_true))


@vjp_rule(func=mae_loss)
def _vjp_mae_loss(g, y_pred, y_true):
    N = y_pred.size
    grad = g * backend.sign(y_pred - y_true) / N
    return (grad, -grad)


# ---------------------------------------------------------------------------
# 3. Huber Loss  (Smooth L1)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def huber_loss(y_pred, y_true, delta=1.0):
    """Huber (smooth L1) loss."""
    r = y_pred - y_true
    abs_r = backend.abs(r)
    quadratic = 0.5 * r * r
    linear = delta * (abs_r - 0.5 * delta)
    return backend.mean(backend.where(abs_r <= delta, quadratic, linear))


@vjp_rule(func=huber_loss)
def _vjp_huber_loss(g, y_pred, y_true, delta=1.0):
    N = y_pred.size
    r = y_pred - y_true
    abs_r = backend.abs(r)
    # Quadratic branch: ∂/∂ŷ = r/N
    # Linear  branch : ∂/∂ŷ = δ·sign(r)/N
    grad = g * backend.where(abs_r <= delta, r, delta * backend.sign(r)) / N
    return (grad, -grad)


# ---------------------------------------------------------------------------
# 4. Log-Cosh Loss
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def log_cosh_loss(y_pred, y_true):
    """Log-cosh loss: mean(log(cosh(ŷ - y)))."""
    r = y_pred - y_true
    # Stable: log cosh(r) = |r| + softplus(-2|r|) - log2
    abs_r = backend.abs(r)
    val = abs_r + backend.log1p(backend.exp(-2.0 * abs_r)) - _LOG2
    return backend.mean(val)


@vjp_rule(func=log_cosh_loss)
def _vjp_log_cosh_loss(g, y_pred, y_true):
    N = y_pred.size
    grad = g * backend.tanh(y_pred - y_true) / N
    return (grad, -grad)


# ---------------------------------------------------------------------------
# 5. Binary Cross-Entropy  (from probabilities)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def bce_loss(y_pred, y_true):
    """Binary cross-entropy from probabilities ∈ (0,1)."""
    eps = safe_eps(y_pred)
    p = backend.clip(y_pred, eps, 1.0 - eps)
    return -backend.mean(
        y_true * backend.log(p) + (1.0 - y_true) * backend.log(1.0 - p)
    )


@vjp_rule(func=bce_loss)
def _vjp_bce_loss(g, y_pred, y_true):
    N = y_pred.size
    eps = safe_eps(y_pred)
    p = backend.clip(y_pred, eps, 1.0 - eps)
    # (p - y) / (p·(1-p)·N)
    grad = g * (p - y_true) / (p * (1.0 - p) * N)
    return (grad, None)  # no gradient w.r.t. targets


# ---------------------------------------------------------------------------
# 6. Binary Cross-Entropy from Logits  (numerically stable)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def bce_with_logits_loss(logits, y_true):
    """Binary cross-entropy directly from logits (numerically stable)."""
    # max(x,0) - x*y + log(1+exp(-|x|))
    relu_logits = backend.maximum(logits, 0.0)
    return backend.mean(
        relu_logits - logits * y_true + backend.log1p(backend.exp(-backend.abs(logits)))
    )


@vjp_rule(func=bce_with_logits_loss)
def _vjp_bce_with_logits_loss(g, logits, y_true):
    N = logits.size
    # Exact: sigmoid(x) - y, never NaN/inf regardless of logit magnitude
    grad = g * (sigmoid(logits) - y_true) / N
    return (grad, None)


# ---------------------------------------------------------------------------
# 7. Categorical Cross-Entropy  (from probability vectors)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def cce_loss(y_pred, y_true):
    """Categorical cross-entropy from probability vectors."""
    eps = safe_eps(y_pred)
    p = backend.clip(y_pred, eps, 1.0)
    return -backend.mean(backend.sum(y_true * backend.log(p), axis=-1))


@vjp_rule(func=cce_loss)
def _vjp_cce_loss(g, y_pred, y_true):
    N = y_pred.shape[0]  # batch size (mean over samples, sum over classes)
    eps = safe_eps(y_pred)
    p = backend.clip(y_pred, eps, 1.0)
    grad = g * (-y_true / (p * N))
    return (grad, None)


# ---------------------------------------------------------------------------
# 8. Categorical Cross-Entropy from Logits  (log-softmax, numerically stable)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def _log_softmax(x, axis=-1):
    """Numerically stable log-softmax."""
    x_max = backend.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    return shifted - backend.log(
        backend.sum(backend.exp(shifted), axis=axis, keepdims=True)
    )


def cce_with_logits_loss(logits, y_true):
    """Categorical cross-entropy directly from logits (log-softmax stable)."""
    log_p = _log_softmax(logits, axis=-1)
    return -backend.mean(backend.sum(y_true * log_p, axis=-1))


@vjp_rule(func=cce_with_logits_loss)
def _vjp_cce_with_logits_loss(g, logits, y_true):
    N = logits.shape[0]
    # Exact: (softmax(x) - y) / N
    grad = g * (softmax(logits, axis=-1) - y_true) / N
    return (grad, None)


# ---------------------------------------------------------------------------
# 9. Sparse Categorical Cross-Entropy from Logits
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def sparse_cce_with_logits_loss(logits, y_true):
    """Sparse categorical cross-entropy from logits (integer targets)."""
    log_p = _log_softmax(logits, axis=-1)
    N = logits.shape[0]
    # Gather log_p at the true class for each sample: log_p[n, y_true[n]]
    idx = (backend.arange(N, dtype=backend.int64), y_true.astype(backend.int64))
    return -backend.mean(log_p[idx])


@vjp_rule(func=sparse_cce_with_logits_loss)
def _vjp_sparse_cce_with_logits_loss(g, logits, y_true):
    N, C = logits.shape
    s = softmax(logits, axis=-1).copy()  # (N, C) — contiguous for scatter
    # Subtract 1 at the true-class position: equivalent to (s - one_hot(y))
    backend.add.at(
        s, (backend.arange(N, dtype=backend.int64), y_true.astype(backend.int64)), -1.0
    )
    return (g * s / N, None)


# ---------------------------------------------------------------------------
# 10. Negative Log-Likelihood Loss  (NLL)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def nll_loss(log_probs, y_true):
    """Negative log-likelihood: -mean(log_probs[n, y[n]])."""
    N = log_probs.shape[0]
    idx = (backend.arange(N, dtype=backend.int64), y_true.astype(backend.int64))
    return -backend.mean(log_probs[idx])


@vjp_rule(func=nll_loss)
def _vjp_nll_loss(g, log_probs, y_true):
    N = log_probs.shape[0]
    grad = backend.zeros_like(log_probs)
    # Scatter -1/N into the true-class positions
    backend.add.at(
        grad,
        (backend.arange(N, dtype=backend.int64), y_true.astype(backend.int64)),
        -1.0 / N,
    )
    return (g * grad, None)


# ---------------------------------------------------------------------------
# 11. KL Divergence  KL(P ‖ Q)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def kl_divergence_loss(p, q):
    """KL divergence KL(p ‖ q) = mean(sum(p · log(p/q), axis=-1))."""
    eps = safe_eps(p)
    p_safe = backend.clip(p, eps, 1.0)
    q_safe = backend.clip(q, eps, 1.0)
    return backend.mean(
        backend.sum(p_safe * (backend.log(p_safe) - backend.log(q_safe)), axis=-1)
    )


@vjp_rule(func=kl_divergence_loss)
def _vjp_kl_divergence_loss(g, p, q):
    # For 2D (B, C): mean is over B, so divide by B.
    # For 1D (C,): sum(axis=-1) → scalar, mean(scalar) = scalar, no division.
    N = p.shape[0] if p.ndim > 1 else 1
    eps = safe_eps(p)
    p_safe = backend.clip(p, eps, 1.0)
    q_safe = backend.clip(q, eps, 1.0)
    grad_p = g * (backend.log(p_safe) - backend.log(q_safe) + 1.0) / N
    grad_q = g * (-p_safe / (q_safe * N))
    return (grad_p, grad_q)


# ---------------------------------------------------------------------------
# 12. Focal Loss  (binary, from logits)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def focal_loss(logits, y_true, gamma=2.0, alpha=0.25):
    """Binary focal loss from logits (Lin et al. 2017)."""
    p = sigmoid(logits)
    # pt: probability of the *true* class
    pt = backend.where(y_true == 1, p, 1.0 - p)
    alpha_t = backend.where(y_true == 1, alpha, 1.0 - alpha)
    # Stable log(pt): use log-sigmoid trick
    log_pt = backend.where(
        y_true == 1,
        -backend.log1p(backend.exp(-backend.abs(logits)))
        - backend.maximum(-logits, 0.0),
        -backend.log1p(backend.exp(-backend.abs(logits)))
        - backend.maximum(logits, 0.0),
    )
    return backend.mean(alpha_t * (1.0 - pt) ** gamma * (-log_pt))


@vjp_rule(func=focal_loss)
def _vjp_focal_loss(g, logits, y_true, gamma=2.0, alpha=0.25):
    N = logits.size
    p = sigmoid(logits)
    pt = backend.where(y_true == 1, p, 1.0 - p)
    alpha_t = backend.where(y_true == 1, alpha, 1.0 - alpha)
    # sign flips because ∂pt/∂x = σ(1-σ) for y=1, -σ(1-σ) for y=0
    sign = backend.where(y_true == 1, 1.0, -1.0)
    # ∂pt/∂x = sign · p · (1-p)
    dpt_dx = sign * p * (1.0 - p)
    # Exact product-rule gradient:
    # d/dx [ (1-pt)^γ · (-log pt) ] =
    #   -γ(1-pt)^(γ-1)·(-dpt_dx)·(-log pt) + (1-pt)^γ·(-dpt_dx/pt)
    #   = (1-pt)^(γ-1) · dpt_dx · [ γ·log(pt) - (1-pt)/pt ]
    # (with appropriate clipping to avoid 0^(γ-1) at pt=1)
    one_minus_pt = backend.clip(1.0 - pt, safe_eps(pt), 1.0)
    pt_safe = backend.clip(pt, safe_eps(pt), 1.0)
    modulating = one_minus_pt ** backend.where(
        backend.array(gamma) >= 1.0, gamma - 1.0, 0.0
    )
    grad = (
        g
        * alpha_t
        * modulating
        * dpt_dx
        * (gamma * backend.log(pt_safe) - one_minus_pt / pt_safe)
        / N
    )
    return (grad, None)


# ---------------------------------------------------------------------------
# 13. Hinge Loss  (multi-class, SVM-style)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def hinge_loss(y_pred, y_true):
    """Hinge loss: mean(max(0, 1 - y·ŷ)).  y ∈ {-1, +1}."""
    return backend.mean(backend.maximum(0.0, 1.0 - y_true * y_pred))


@vjp_rule(func=hinge_loss)
def _vjp_hinge_loss(g, y_pred, y_true):
    N = y_pred.size
    # Indicator: 1 where margin is violated
    mask = (y_true * y_pred < 1.0).astype(y_pred.dtype)
    grad = g * (-y_true * mask) / N
    return (grad, None)


# ---------------------------------------------------------------------------
# 14. Squared Hinge Loss
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def squared_hinge_loss(y_pred, y_true):
    """Squared hinge loss: mean(max(0, 1 - y·ŷ)²)."""
    h = backend.maximum(0.0, 1.0 - y_true * y_pred)
    return backend.mean(h * h)


@vjp_rule(func=squared_hinge_loss)
def _vjp_squared_hinge_loss(g, y_pred, y_true):
    N = y_pred.size
    h = backend.maximum(0.0, 1.0 - y_true * y_pred)
    grad = g * (-2.0 * y_true * h) / N
    return (grad, None)


# ---------------------------------------------------------------------------
# 15. Cosine Embedding Loss
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def cosine_embedding_loss(u, v, y, margin=0.0):
    """Cosine embedding loss for similarity/dissimilarity pairs."""
    eps = safe_eps(u)
    nu = backend.sqrt(backend.sum(u * u, axis=-1, keepdims=True)).clip(eps)
    nv = backend.sqrt(backend.sum(v * v, axis=-1, keepdims=True)).clip(eps)
    cos_sim = backend.sum(u * v, axis=-1) / (nu.squeeze(-1) * nv.squeeze(-1))
    loss_pos = 1.0 - cos_sim
    loss_neg = backend.maximum(0.0, cos_sim - margin)
    per_sample = backend.where(y == 1, loss_pos, loss_neg)
    return backend.mean(per_sample)


@vjp_rule(func=cosine_embedding_loss)
def _vjp_cosine_embedding_loss(g, u, v, y, margin=0.0):
    N = u.shape[0]
    eps = safe_eps(u)
    nu = backend.sqrt(backend.sum(u * u, axis=-1, keepdims=True)).clip(eps)  # (N,1)
    nv = backend.sqrt(backend.sum(v * v, axis=-1, keepdims=True)).clip(eps)
    cos_sim = backend.sum(u * v, axis=-1) / (nu.squeeze(-1) * nv.squeeze(-1))  # (N,)

    # ∂c/∂u_i = v/(nu·nv) - c·u/nu²  — shaped (N, D)
    nu_nv = nu * nv  # (N,1)
    c = cos_sim[:, backend.newaxis]  # (N,1) broadcast
    dc_du = (v / nu_nv) - (c * u / (nu * nu))
    dc_dv = (u / nu_nv) - (c * v / (nv * nv))

    # ∂L_i/∂c_i  (scalar per sample)
    dl_dc = backend.where(
        y == 1,
        -backend.ones_like(cos_sim),
        backend.where(
            cos_sim > margin, backend.ones_like(cos_sim), backend.zeros_like(cos_sim)
        ),
    )[:, backend.newaxis]  # (N,1)

    grad_u = g * dl_dc * dc_du / N
    grad_v = g * dl_dc * dc_dv / N
    return (grad_u, grad_v, None)  # no grad w.r.t. y


# ---------------------------------------------------------------------------
# 16. Triplet Margin Loss
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """Triplet margin loss: mean(max(0, d(a,p) - d(a,n) + margin))."""
    eps = safe_eps(anchor)
    d_pos = backend.sqrt(backend.sum((anchor - positive) ** 2, axis=-1) + eps)
    d_neg = backend.sqrt(backend.sum((anchor - negative) ** 2, axis=-1) + eps)
    return backend.mean(backend.maximum(0.0, d_pos - d_neg + margin))


@vjp_rule(func=triplet_margin_loss)
def _vjp_triplet_margin_loss(g, anchor, positive, negative, margin=1.0):
    N = anchor.shape[0]
    eps = safe_eps(anchor)
    d_pos = backend.sqrt(
        backend.sum((anchor - positive) ** 2, axis=-1, keepdims=True) + eps
    )
    d_neg = backend.sqrt(
        backend.sum((anchor - negative) ** 2, axis=-1, keepdims=True) + eps
    )
    # Active mask: samples where loss > 0
    active = ((d_pos.squeeze(-1) - d_neg.squeeze(-1) + margin) > 0.0).astype(
        anchor.dtype
    )[:, backend.newaxis]  # (N,1)

    diff_pos = anchor - positive  # (N,D)
    diff_neg = anchor - negative

    grad_a = g * active * (diff_pos / d_pos - diff_neg / d_neg) / N
    grad_p = g * active * (-diff_pos / d_pos) / N
    grad_n = g * active * (diff_neg / d_neg) / N
    return (grad_a, grad_p, grad_n)


# ---------------------------------------------------------------------------
# 17. Dice Loss  (binary segmentation)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def dice_loss(y_pred, y_true, eps=1.0):
    """Dice loss for binary segmentation: 1 - Dice coefficient.
    eps=1.0 (Laplace smoothing) prevents division by zero on empty masks.
    """
    # Flatten all spatial dims per sample; keep batch dim
    flat_pred = y_pred.reshape(y_pred.shape[0], -1)
    flat_true = y_true.reshape(y_true.shape[0], -1)
    intersection = backend.sum(flat_pred * flat_true, axis=1)
    sum_pred = backend.sum(flat_pred, axis=1)
    sum_true = backend.sum(flat_true, axis=1)
    dice = (2.0 * intersection + eps) / (sum_pred + sum_true + eps)
    return backend.mean(1.0 - dice)


@vjp_rule(func=dice_loss)
def _vjp_dice_loss(g, y_pred, y_true, eps=1.0):
    N = y_pred.shape[0]
    flat_pred = y_pred.reshape(N, -1)
    flat_true = y_true.reshape(N, -1)
    intersection = backend.sum(flat_pred * flat_true, axis=1, keepdims=True)  # (N,1)
    sum_pred = backend.sum(flat_pred, axis=1, keepdims=True)
    sum_true = backend.sum(flat_true, axis=1, keepdims=True)
    denom = sum_pred + sum_true + eps  # (N,1)
    numer = 2.0 * intersection + eps  # (N,1)  = T in derivation
    # Quotient rule on dice = T/D:
    #   ∂dice/∂ŷ_i = (∂T/∂ŷ_i · D - T · ∂D/∂ŷ_i) / D²
    #             = (2·y_i · D  -  T · 1) / D²
    # L = 1 - dice  →  ∂L/∂ŷ_i = -(2·y_i·D - T) / D²  / N
    grad_flat = g * (-(2.0 * flat_true * denom - numer) / (denom * denom)) / N
    return (grad_flat.reshape(y_pred.shape), None)


# ---------------------------------------------------------------------------
# 18. Tversky Loss  (generalised Dice with α/β asymmetry)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def tversky_loss(y_pred, y_true, alpha=0.3, beta=0.7, eps=1.0):
    """Tversky loss: 1 - Tversky index.  α controls FP, β controls FN penalty."""
    N = y_pred.shape[0]
    flat_pred = y_pred.reshape(N, -1)
    flat_true = y_true.reshape(N, -1)
    TP = backend.sum(flat_pred * flat_true, axis=1)
    FP = backend.sum(flat_pred * (1.0 - flat_true), axis=1)
    FN = backend.sum((1.0 - flat_pred) * flat_true, axis=1)
    tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    return backend.mean(1.0 - tversky)


@vjp_rule(func=tversky_loss)
def _vjp_tversky_loss(g, y_pred, y_true, alpha=0.3, beta=0.7, eps=1.0):
    N = y_pred.shape[0]
    flat_pred = y_pred.reshape(N, -1)
    flat_true = y_true.reshape(N, -1)

    TP = backend.sum(flat_pred * flat_true, axis=1, keepdims=True)  # (N,1)
    FP = backend.sum(flat_pred * (1.0 - flat_true), axis=1, keepdims=True)
    FN = backend.sum((1.0 - flat_pred) * flat_true, axis=1, keepdims=True)

    D = TP + alpha * FP + beta * FN + eps  # denominator
    T = TP + eps  # numerator

    # Per-element derivatives
    dT_dpred = flat_true  # ∂TP/∂ŷ
    dD_dpred = flat_true + alpha * (1.0 - flat_true) - beta * flat_true
    #         = ∂TP/∂ŷ + α·∂FP/∂ŷ + β·∂FN/∂ŷ
    #         = y + α(1-y) - β·y

    # Quotient rule: ∂tversky/∂ŷ = (dT·D - T·dD) / D²
    # ∂(1-tversky)/∂ŷ = -(dT·D - T·dD) / D²
    grad_flat = g * (-(dT_dpred * D - T * dD_dpred) / (D * D)) / N
    return (grad_flat.reshape(y_pred.shape), None)


# ---------------------------------------------------------------------------
# 19. Wasserstein Loss  (WGAN critic objective)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def wasserstein_loss(scores_real, scores_fake):
    """WGAN critic loss: mean(fake) - mean(real).  Minimise for the critic."""
    return backend.mean(scores_fake) - backend.mean(scores_real)


@vjp_rule(func=wasserstein_loss)
def _vjp_wasserstein_loss(g, scores_real, scores_fake):
    N_real = scores_real.size
    N_fake = scores_fake.size
    grad_real = g * backend.full_like(scores_real, -1.0 / N_real)
    grad_fake = g * backend.full_like(scores_fake, +1.0 / N_fake)
    return (grad_real, grad_fake)


# ---------------------------------------------------------------------------
# 20. SSIM Loss  (Structural Similarity Index, patch-level)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def ssim_loss(x, y, k1=0.01, k2=0.03, L=1.0):
    """1 - SSIM.  x, y: (N, H, W) or (N, C, H, W), values in [0,1]."""
    # Flatten spatial dims: work on (N, P) where P=C*H*W or H*W
    N = x.shape[0]
    xf = x.reshape(N, -1).astype(
        backend.float64 if x.dtype == backend.float32 else x.dtype
    )
    yf = y.reshape(N, -1).astype(xf.dtype)
    P = xf.shape[1]

    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    mu_x = backend.mean(xf, axis=1, keepdims=True)  # (N,1)
    mu_y = backend.mean(yf, axis=1, keepdims=True)
    dx = xf - mu_x
    dy = yf - mu_y
    var_x = backend.mean(dx * dx, axis=1)  # (N,)
    var_y = backend.mean(dy * dy, axis=1)
    cov_xy = backend.mean(dx * dy, axis=1)

    A = 2.0 * mu_x.squeeze(1) * mu_y.squeeze(1) + c1
    B = 2.0 * cov_xy + c2
    C = mu_x.squeeze(1) ** 2 + mu_y.squeeze(1) ** 2 + c1
    D = var_x + var_y + c2

    ssim_map = (A * B) / (C * D)
    return backend.mean(1.0 - ssim_map).astype(x.dtype)


@vjp_rule(func=ssim_loss)
def _vjp_ssim_loss(g, x, y, k1=0.01, k2=0.03, L=1.0):
    N = x.shape[0]
    orig_shape = x.shape
    xf = x.reshape(N, -1).astype(
        backend.float64 if x.dtype == backend.float32 else x.dtype
    )
    yf = y.reshape(N, -1).astype(xf.dtype)
    P = xf.shape[1]

    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    mu_x = backend.mean(xf, axis=1, keepdims=True)  # (N,1)
    mu_y = backend.mean(yf, axis=1, keepdims=True)
    dx = xf - mu_x
    dy = yf - mu_y
    var_x = backend.mean(dx * dx, axis=1)  # (N,)
    var_y = backend.mean(dy * dy, axis=1)
    cov_xy = backend.mean(dx * dy, axis=1)

    A = 2.0 * mu_x.squeeze(1) * mu_y.squeeze(1) + c1  # (N,)
    B = 2.0 * cov_xy + c2
    C = mu_x.squeeze(1) ** 2 + mu_y.squeeze(1) ** 2 + c1
    D = var_x + var_y + c2
    CD = (C * D)[:, backend.newaxis]  # (N,1)
    CD2 = CD * CD

    # ∂SSIM/∂x_i via quotient rule (all quantities shaped (N,P) after broadcast)
    # ∂A/∂x_i = 2·μ_y / P   →  (N,1)
    dA = 2.0 * mu_y / P  # (N,1) — same for all pixels
    # ∂B/∂x_i = 2(y_i - μ_y) / P
    dB = 2.0 * dy / P  # (N,P)
    # ∂C/∂x_i = 2μ_x / P   →  (N,1)
    dC = 2.0 * mu_x / P
    # ∂D/∂x_i = 2(x_i - μ_x) / P
    dD = 2.0 * dx / P  # (N,P)

    A_ = A[:, backend.newaxis]
    B_ = B[:, backend.newaxis]
    C_ = C[:, backend.newaxis]
    D_ = D[:, backend.newaxis]

    # d(SSIM)/dx_i  =  [dA·B·C·D + A·dB·C·D - A·B·dC·D - A·B·C·dD] / (C·D)²
    num_grad = (
        dA * B_ * C_ * D_ + A_ * dB * C_ * D_ - A_ * B_ * dC * D_ - A_ * B_ * C_ * dD
    )
    # L = 1 - mean(SSIM) → ∂L/∂x = -∂SSIM/∂x / N
    grad_flat = g * (-num_grad / CD2) / N
    return (grad_flat.reshape(orig_shape).astype(x.dtype), None)


# ===========================================================================
# Recurrent & Sequence VJP rules  (from VJP_RULES_RECURRENT.md)
# ===========================================================================
#
# Design contract
# ---------------
# * Every forward function receives and returns plain arrays, not Tensors.
# * Caches needed for the backward (intermediate activations) are re-computed
#   inside each VJP from the same forward call, keeping the API stateless.
# * All ops are fully backend-agnostic — only backend.* calls, no np/cp.
# * BPTT is handled inside each VJP; the compute graph sees one node per layer.


# ---------------------------------------------------------------------------
# 1. Simple (Elman) RNN
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def _rnn_forward_cache(x, Wh, Wx, bh, h0):
    """Run forward pass and return (h_seq, h0) for BPTT."""
    B, T, _ = x.shape
    d_h = Wh.shape[0]
    if h0 is None:
        h0 = backend.zeros((B, d_h), dtype=x.dtype)
    h_seq = backend.zeros((B, T, d_h), dtype=x.dtype)
    h_prev = h0
    for t in range(T):
        h_t = backend.tanh(h_prev @ Wh + x[:, t, :] @ Wx + bh)
        h_seq[:, t, :] = h_t
        h_prev = h_t
    return h_seq, h0


def rnn_cell(x, Wh, Wx, bh, h0=None):
    """Simple Elman RNN forward over a full sequence → h_seq (B, T, d_h)."""
    h_seq, _ = _rnn_forward_cache(x, Wh, Wx, bh, h0)
    return h_seq


@vjp_rule(func=rnn_cell)
def _vjp_rnn_cell(g, x, Wh, Wx, bh, h0=None, h_seq=None, h0_used=None):
    """BPTT VJP for Simple RNN.  g : (B, T, d_h).  Returns (dx, dWh, dWx, dbh, dh0)."""
    # P2: reuse cached forward activations if available; otherwise recompute.
    if h_seq is None:
        h_seq, h0_used = _rnn_forward_cache(x, Wh, Wx, bh, h0)
    B, T, _ = x.shape
    d_h = Wh.shape[0]
    dx = backend.zeros_like(x)
    dWh = backend.zeros_like(Wh)
    dWx = backend.zeros_like(Wx)
    dbh = backend.zeros_like(bh)
    dh_next = backend.zeros((B, d_h), dtype=x.dtype)
    for t in reversed(range(T)):
        h_t = h_seq[:, t, :]
        h_prev = h_seq[:, t - 1, :] if t > 0 else h0_used
        delta = (g[:, t, :] + dh_next) * (1.0 - h_t**2)
        dWh += h_prev.T @ delta
        dWx += x[:, t, :].T @ delta
        dbh += delta.sum(axis=0)
        dh_next = delta @ Wh.T
        dx[:, t, :] = delta @ Wx.T
    return (dx, dWh, dWx, dbh, dh_next)


# ---------------------------------------------------------------------------
# 2. LSTM
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def _lstm_forward_cache(x, W, b, h0, c0):
    B, T, d_in = x.shape
    d_h = W.shape[1] // 4
    if h0 is None:
        h0 = backend.zeros((B, d_h), dtype=x.dtype)
    if c0 is None:
        c0 = backend.zeros((B, d_h), dtype=x.dtype)
    h_seq = backend.zeros((B, T, d_h), dtype=x.dtype)
    c_seq = backend.zeros((B, T, d_h), dtype=x.dtype)
    f_seq = backend.zeros((B, T, d_h), dtype=x.dtype)
    i_seq = backend.zeros((B, T, d_h), dtype=x.dtype)
    ct_seq = backend.zeros((B, T, d_h), dtype=x.dtype)
    o_seq = backend.zeros((B, T, d_h), dtype=x.dtype)
    h_prev, c_prev = h0, c0
    for t in range(T):
        z_t = backend.concatenate([h_prev, x[:, t, :]], axis=1)
        gs = z_t @ W + b
        f_t = sigmoid(gs[:, :d_h])
        i_t = sigmoid(gs[:, d_h : 2 * d_h])
        ct_t = backend.tanh(gs[:, 2 * d_h : 3 * d_h])
        o_t = sigmoid(gs[:, 3 * d_h :])
        c_t = f_t * c_prev + i_t * ct_t
        h_t = o_t * backend.tanh(c_t)
        h_seq[:, t, :] = h_t
        c_seq[:, t, :] = c_t
        f_seq[:, t, :] = f_t
        i_seq[:, t, :] = i_t
        ct_seq[:, t, :] = ct_t
        o_seq[:, t, :] = o_t
        h_prev, c_prev = h_t, c_t
    return h_seq, c_seq, f_seq, i_seq, ct_seq, o_seq, h0, c0


def lstm_cell(x, W, b, h0=None, c0=None):
    """LSTM forward over a full sequence → h_seq (B, T, d_h)."""
    h_seq, *_ = _lstm_forward_cache(x, W, b, h0, c0)
    return h_seq


@vjp_rule(func=lstm_cell)
def _vjp_lstm_cell(
    g,
    x,
    W,
    b,
    h0=None,
    c0=None,
    h_seq=None,
    c_seq=None,
    f_seq=None,
    i_seq=None,
    ct_seq=None,
    o_seq=None,
    h0_u=None,
    c0_u=None,
):
    """BPTT VJP for LSTM.  g : (B, T, d_h).  Returns (dx, dW, db, dh0, dc0)."""
    # P2: reuse cached forward activations if available; otherwise recompute.
    if h_seq is None:
        h_seq, c_seq, f_seq, i_seq, ct_seq, o_seq, h0_u, c0_u = _lstm_forward_cache(
            x, W, b, h0, c0
        )
    B, T, _ = x.shape
    d_h = W.shape[1] // 4
    dx = backend.zeros_like(x)
    dW = backend.zeros_like(W)
    db = backend.zeros_like(b)
    dh_next = backend.zeros((B, d_h), dtype=x.dtype)
    dc_next = backend.zeros((B, d_h), dtype=x.dtype)
    for t in reversed(range(T)):
        c_t = c_seq[:, t, :]
        f_t = f_seq[:, t, :]
        i_t = i_seq[:, t, :]
        ct_t = ct_seq[:, t, :]
        o_t = o_seq[:, t, :]
        c_prev = c_seq[:, t - 1, :] if t > 0 else c0_u
        h_prev = h_seq[:, t - 1, :] if t > 0 else h0_u
        dh_t = g[:, t, :] + dh_next
        d_ot = dh_t * backend.tanh(c_t)
        dc_total = dc_next + dh_t * o_t * (1.0 - backend.tanh(c_t) ** 2)
        d_ft = dc_total * c_prev
        d_it = dc_total * ct_t
        d_ctt = dc_total * i_t
        dc_next = dc_total * f_t
        delta_f = d_ft * f_t * (1.0 - f_t)
        delta_i = d_it * i_t * (1.0 - i_t)
        delta_ct = d_ctt * (1.0 - ct_t**2)
        delta_o = d_ot * o_t * (1.0 - o_t)
        Delta = backend.concatenate([delta_f, delta_i, delta_ct, delta_o], axis=1)
        z_t = backend.concatenate([h_prev, x[:, t, :]], axis=1)
        dW += z_t.T @ Delta
        db += Delta.sum(axis=0)
        dz = Delta @ W.T
        dh_next = dz[:, :d_h]
        dx[:, t, :] = dz[:, d_h:]
    return (dx, dW, db, dh_next, dc_next)


# ---------------------------------------------------------------------------
# 3. GRU
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def _gru_forward_cache(x, Wr, Wz, Wh, br, bz, bh, h0):
    B, T, _ = x.shape
    d_h = Wr.shape[1]
    if h0 is None:
        h0 = backend.zeros((B, d_h), dtype=x.dtype)
    h_seq = backend.zeros((B, T, d_h), dtype=x.dtype)
    r_seq = backend.zeros((B, T, d_h), dtype=x.dtype)
    z_seq = backend.zeros((B, T, d_h), dtype=x.dtype)
    ht_seq = backend.zeros((B, T, d_h), dtype=x.dtype)
    h_prev = h0
    for t in range(T):
        xh = backend.concatenate([h_prev, x[:, t, :]], axis=1)
        r_t = sigmoid(xh @ Wr + br)
        z_t = sigmoid(xh @ Wz + bz)
        rh = backend.concatenate([r_t * h_prev, x[:, t, :]], axis=1)
        ht_t = backend.tanh(rh @ Wh + bh)
        h_t = (1.0 - z_t) * h_prev + z_t * ht_t
        h_seq[:, t, :] = h_t
        r_seq[:, t, :] = r_t
        z_seq[:, t, :] = z_t
        ht_seq[:, t, :] = ht_t
        h_prev = h_t
    return h_seq, r_seq, z_seq, ht_seq, h0


def gru_cell(x, Wr, Wz, Wh, br, bz, bh, h0=None):
    """GRU forward over a full sequence → h_seq (B, T, d_h)."""
    h_seq, *_ = _gru_forward_cache(x, Wr, Wz, Wh, br, bz, bh, h0)
    return h_seq


@vjp_rule(func=gru_cell)
def _vjp_gru_cell(
    g,
    x,
    Wr,
    Wz,
    Wh,
    br,
    bz,
    bh,
    h0=None,
    h_seq=None,
    r_seq=None,
    z_seq=None,
    ht_seq=None,
    h0_u=None,
):
    """BPTT VJP for GRU.  Returns (dx, dWr, dWz, dWh, dbr, dbz, dbh, dh0)."""
    # P2: reuse cached forward activations if available; otherwise recompute.
    if h_seq is None:
        h_seq, r_seq, z_seq, ht_seq, h0_u = _gru_forward_cache(
            x, Wr, Wz, Wh, br, bz, bh, h0
        )
    B, T, _ = x.shape
    d_h = Wr.shape[1]
    dx = backend.zeros_like(x)
    dWr = backend.zeros_like(Wr)
    dWz = backend.zeros_like(Wz)
    dWh = backend.zeros_like(Wh)
    dbr = backend.zeros_like(br)
    dbz = backend.zeros_like(bz)
    dbh = backend.zeros_like(bh)
    dh_next = backend.zeros((B, d_h), dtype=x.dtype)
    for t in reversed(range(T)):
        r_t = r_seq[:, t, :]
        z_t = z_seq[:, t, :]
        ht_t = ht_seq[:, t, :]
        h_prev = h_seq[:, t - 1, :] if t > 0 else h0_u
        dh_t = g[:, t, :] + dh_next
        # Step 1 — hidden update
        d_htt = dh_t * z_t
        d_zt = dh_t * (ht_t - h_prev)
        dh_prev1 = dh_t * (1.0 - z_t)
        # Step 2 — candidate h̃_t
        delta_ht = d_htt * (1.0 - ht_t**2)
        rh = backend.concatenate([r_t * h_prev, x[:, t, :]], axis=1)
        dWh += rh.T @ delta_ht
        dbh += delta_ht.sum(axis=0)
        d_rh = delta_ht @ Wh.T
        d_roh = d_rh[:, :d_h]
        dx_ht = d_rh[:, d_h:]
        # Step 3 — reset gate multiplication
        d_rt = d_roh * h_prev
        dh_prev2 = d_roh * r_t
        # Step 4 — reset gate sigmoid
        xh = backend.concatenate([h_prev, x[:, t, :]], axis=1)
        delta_r = d_rt * r_t * (1.0 - r_t)
        dWr += xh.T @ delta_r
        dbr += delta_r.sum(axis=0)
        d_xh_r = delta_r @ Wr.T
        dh_prev3 = d_xh_r[:, :d_h]
        dx_r = d_xh_r[:, d_h:]
        # Step 5 — update gate sigmoid
        delta_z = d_zt * z_t * (1.0 - z_t)
        dWz += xh.T @ delta_z
        dbz += delta_z.sum(axis=0)
        d_xh_z = delta_z @ Wz.T
        dh_prev4 = d_xh_z[:, :d_h]
        dx_z = d_xh_z[:, d_h:]
        # Step 6 — accumulate
        dh_next = dh_prev1 + dh_prev2 + dh_prev3 + dh_prev4
        dx[:, t, :] = dx_ht + dx_r + dx_z
    return (dx, dWr, dWz, dWh, dbr, dbz, dbh, dh_next)


# ---------------------------------------------------------------------------
# 4. Layer Normalization
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization: y = γ·norm(x) + β, normalized over last axis."""
    mu = backend.mean(x, axis=-1, keepdims=True)
    var = backend.mean((x - mu) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mu) / backend.sqrt(var + eps)
    return gamma * x_hat + beta


@vjp_rule(func=layer_norm)
def _vjp_layer_norm(g, x, gamma, beta, eps=1e-5):
    """Closed-form VJP for LayerNorm.  Returns (dx, dgamma, dbeta)."""
    mu = backend.mean(x, axis=-1, keepdims=True)
    var = backend.mean((x - mu) ** 2, axis=-1, keepdims=True)
    std = backend.sqrt(var + eps)
    x_hat = (x - mu) / std
    D = x.shape[-1]
    g_x_hat = g * gamma
    # Reduce over all axes except the last (the normalised dim)
    reduce_axes = tuple(range(x.ndim - 1))
    dgamma = (g * x_hat).sum(axis=reduce_axes)
    dbeta = g.sum(axis=reduce_axes)
    dx = (1.0 / (D * std)) * (
        D * g_x_hat
        - backend.sum(g_x_hat, axis=-1, keepdims=True)
        - x_hat * backend.sum(g_x_hat * x_hat, axis=-1, keepdims=True)
    )
    return (dx, dgamma, dbeta)


# ---------------------------------------------------------------------------
# 5. Gather / Embedding Lookup
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def gather(table, idx):
    """Gather rows of *table* at integer indices *idx* (any shape).

    Parameters
    ----------
    table : (V, d) weight matrix
    idx   : integer index array, any shape

    Returns
    -------
    y : (*idx.shape, d)
    """
    if hasattr(idx, "data"):
        idx = idx.data
    idx_array = backend.asarray(idx, dtype=int)
    return table[idx_array]


# Legacy alias kept for backwards compatibility.
embedding_lookup = gather


def embedding(table, idx):
    return gather(table, idx)


@vjp_rule(func=embedding)
def _vjp_embedding(g, table, idx):
    return _vjp_gather(g, table, idx)


@vjp_rule(func=gather)
def _vjp_gather(g, table, idx):
    """VJP for gather/embedding — sparse scatter-add.  Returns (d_table, None)."""
    d_table = backend.zeros_like(table)
    backend.add.at(d_table, idx, g)
    return (d_table, None)


# Also register under the old name so any orphaned VJP_RULES lookups still hit.
VJP_RULES[embedding_lookup] = _vjp_gather


# ---------------------------------------------------------------------------
# 5b. General Indexing (getitem) — differentiable slice / fancy-index
# ---------------------------------------------------------------------------


def getitem(x, idx):
    """Return ``x[idx]`` for any valid NumPy index (slice, int, array).\n\n    The index *idx* is treated as non-differentiable; only *x* has a VJP."""
    return x[idx]


@vjp_rule(func=getitem)
def _vjp_getitem(g, x, idx):
    """Scatter gradient back into *x* shape via ``add.at``."""
    x_grad = backend.zeros_like(x)
    backend.add.at(x_grad, idx, g)
    return (x_grad, None)  # None for non-differentiable idx


def setitem(x, idx, value):
    """Differentiable copy-and-set: returns a copy of *x* with ``x[idx] = value``.

    Unlike Tensor.__setitem__ (which mutates *x* in-place and breaks the graph),
    this function creates a new Tensor in the computation graph so gradients
    flow correctly through both *x* and *value*.

    Parameters
    ----------
    x : array-like
        Source tensor.
    idx : slice / int / array
        Index expression — not differentiable.
    value : array-like
        Values to insert at *idx* — differentiable.

    Returns
    -------
    array  shape identical to *x*
    """
    out = x.copy()
    out[idx] = value
    return out


@vjp_rule(func=setitem)
def _vjp_setitem(g, x, idx, value):
    """VJP for the copy-and-set op.

    *  grad_x   = g, but zeroed at the positions written by idx
       (those positions came purely from *value*, not from *x*).
    *  grad_idx = None  (non-differentiable index)
    *  grad_val = g[idx]  (gradient flowing back to the inserted values)
    """
    grad_x = g.copy()
    grad_x[idx] = 0
    grad_val = g[idx]
    return (grad_x, None, grad_val)


# ---------------------------------------------------------------------------
# 6. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention → out (B, N, d_v)."""
    d_k = Q.shape[-1]
    scale = backend.sqrt(backend.array(float(d_k), dtype=Q.dtype))
    E = backend.matmul(Q, backend.swapaxes(K, -1, -2)) / scale
    if mask is not None:
        E = backend.where(mask, backend.full_like(E, -1e9), E)
    A = softmax(E, axis=-1)
    return backend.matmul(A, V)


@vjp_rule(func=scaled_dot_product_attention)
def _vjp_scaled_dot_product_attention(g, Q, K, V, mask=None):
    """Exact VJP for scaled dot-product attention.  Returns (dQ, dK, dV, None)."""
    d_k = Q.shape[-1]
    scale = backend.sqrt(backend.array(float(d_k), dtype=Q.dtype))
    E = backend.matmul(Q, backend.swapaxes(K, -1, -2)) / scale
    if mask is not None:
        E = backend.where(mask, backend.full_like(E, -1e9), E)
    A = softmax(E, axis=-1)

    # Step 1 — dV and dA
    dV = backend.matmul(backend.swapaxes(A, -1, -2), g)  # (B, M, d_v)
    dA = backend.matmul(g, backend.swapaxes(V, -1, -2))  # (B, N, M)

    # Step 2 — softmax VJP: dE = A ⊙ (dA − rowsum(A ⊙ dA))
    dE = A * (dA - backend.sum(A * dA, axis=-1, keepdims=True))
    dE = dE / scale

    # Step 3 — dQ and dK
    dQ = backend.matmul(dE, K)  # (B, N, d_k)
    dK = backend.matmul(backend.swapaxes(dE, -1, -2), Q)  # (B, M, d_k)

    return (dQ, dK, dV, None)


# ---------------------------------------------------------------------------
# 7. Advanced Complex Operations (RoPE, FlashAttention, Sinkhorn, NeuralODE, S4)
# ---------------------------------------------------------------------------

# Note: For the mathematical VJP derivation, see the VJP_RULES*.md documentation.


def rope(x, cos_freqs, sin_freqs):
    """
    Rotary Position Embedding (RoPE) forward pass.
    Assumes the last dimension is split in half for rotation.
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    out1 = x1 * cos_freqs - x2 * sin_freqs
    out2 = x1 * sin_freqs + x2 * cos_freqs
    return backend.concatenate([out1, out2], axis=-1)


@vjp_rule(func=rope)
def _vjp_rope(g, x, cos_freqs, sin_freqs):
    """VJP for RoPE applies the exact same rotation but with a negative angle."""
    # cos(-t) = cos(t), sin(-t) = -sin(t)
    gx = rope(g, cos_freqs, -sin_freqs)
    # No gradients flow to the frequency tensors.
    return (gx, None, None)


def flash_attention(Q, K, V, mask=None):
    """
    Simulated flash-attention forward (standard math, identical output).
    Real hardware flash-attention uses custom CUDA kernels.
    """
    return scaled_dot_product_attention(Q, K, V, mask=mask)


@vjp_rule(func=flash_attention)
def _vjp_flash_attention(g, Q, K, V, mask=None):
    """
    Simulated FlashAttention backward.
    In a true implementation, this recomputes the attention matrix tile-by-tile
    in SRAM to achieve O(1) memory. Here we simulate the mathematical gradients.
    """
    return _vjp_scaled_dot_product_attention(g, Q, K, V, mask=mask)


def sinkhorn(a, b, M, reg, num_iters=20):
    """
    Sinkhorn-Knopp algorithm for Optimal Transport.
    Returns the optimal coupling matrix P.
    """
    K = backend.exp(-M / reg)
    u = backend.ones_like(a) / a.shape[-1]

    # Expand dims for proper batched matrix-vector multiplication
    u = backend.expand_dims(u, -1)
    b_ = backend.expand_dims(b, -1)
    a_ = backend.expand_dims(a, -1)

    for _ in range(num_iters):
        v = b_ / (backend.matmul(backend.swapaxes(K, -1, -2), u) + safe_eps(b_))
        u = a_ / (backend.matmul(K, v) + safe_eps(a_))

    return u * K * backend.swapaxes(v, -1, -2)


@vjp_rule(func=sinkhorn)
def _vjp_sinkhorn(g, a, b, M, reg, num_iters=20):
    """
    VJP for Sinkhorn via Implicit Function Theorem (simulated approximation).
    Avoids unrolling the loop overhead.
    """
    P = sinkhorn(a, b, M, reg, num_iters)
    dM = -(1.0 / reg) * P * (g - backend.sum(g * P, axis=-1, keepdims=True))
    return (None, None, dM, None, None)


def _apply_odefunc_data(odefunc, z):
    """Run a Module odefunc on raw array ``z``, returning the output as a raw array.

    Temporarily disables the computation graph so that internal Tensor creation
    inside ``odefunc`` does not pollute the outer session graph.
    """
    prev = session._grad_enabled
    session._grad_enabled = False
    try:
        z_t = Tensor(z, name="ode_z")
        out = odefunc(z_t)
    finally:
        session._grad_enabled = prev
    return out.data if hasattr(out, "data") else out


def neural_ode_solve(z0, t_span, steps=10, odefunc=None):
    """
    Neural ODE forward solver using the Euler method.
    z(t+dt) = z(t) + f(z) * dt

    Parameters
    ----------
    z0 : array
        Initial state.
    t_span : array of shape (2,)
        ``[t0, t1]`` — start and end times.
    steps : int
        Number of Euler integration steps.
    odefunc : callable (Module), optional
        The ODE function ``f(z) -> dz/dt``.  When *None* a mock
        ``f(z) = -0.5 * z`` is used for backward-compatibility testing.
    """
    dt = (t_span[1] - t_span[0]) / steps
    z = z0
    for _ in range(steps):
        dz = _apply_odefunc_data(odefunc, z) if odefunc is not None else -0.5 * z
        z = z + dz * dt
    return z


@vjp_rule(func=neural_ode_solve)
def _vjp_neural_ode_solve(g, z0, t_span, steps=10, odefunc=None):
    """Continuous adjoint method for Neural ODE.

    Rebuilds the forward trajectory then integrates the adjoint ODE backward.
    When ``odefunc`` is provided the Jacobian-vector product ``J_f^T @ a`` is
    approximated via element-wise finite differences (exact up to O(eps^2)).

    Note: gradients are computed only w.r.t. ``z0``.  Gradients w.r.t.
    odefunc parameters require a separate differentiation pass through the
    module.
    """
    dt = (t_span[1] - t_span[0]) / steps
    EPS_FD = 1e-5

    # --- rebuild forward trajectory ---
    z = z0
    trajectory = [z.copy()]
    for _ in range(steps):
        dz = _apply_odefunc_data(odefunc, z) if odefunc is not None else -0.5 * z
        z = z + dz * dt
        trajectory.append(z.copy())

    # --- adjoint backward pass ---
    a = g.copy()
    for i in range(steps - 1, -1, -1):
        z_i = trajectory[i]
        if odefunc is not None:
            f_z = _apply_odefunc_data(odefunc, z_i)
            # Compute J_f^T @ a via element-wise finite differences.
            # (J_f^T @ a)_i = sum_j (df_j/dz_i) * a_j
            Jt_a = backend.zeros_like(a)
            z_flat = z_i.ravel()
            a_flat = a.ravel()
            for j in range(z_i.size):
                z_pert = z_flat.copy()
                z_pert[j] += EPS_FD
                col_j = (
                    _apply_odefunc_data(odefunc, z_pert.reshape(z_i.shape)).ravel()
                    - f_z.ravel()
                ) / EPS_FD
                Jt_a.ravel()[j] = float(
                    backend.dot(
                        a_flat.astype(backend.float64), col_j.astype(backend.float64)
                    )
                )
            a = a + Jt_a * dt
        else:
            # f(z) = -0.5*z  →  J_f = -0.5*I  →  J_f^T @ a = -0.5*a
            a = a + (-0.5 * a) * dt
    return (a, None, None)


def _s4_forward_cache(u, A, B, C):
    """Run S4 state space forward and return (y_seq, x_seq) for VJP."""
    batch, seq_len, d_in = u.shape
    d_model = A.shape[-1]
    d_out = C.shape[-1]

    # Vectorized pre-allocation
    x_prev = backend.zeros((batch, d_model), dtype=u.dtype)
    x_seq = backend.zeros((batch, seq_len, d_model), dtype=u.dtype)
    y_seq = backend.zeros((batch, seq_len, d_out), dtype=u.dtype)

    for t in range(seq_len):
        x_curr = backend.matmul(x_prev, A) + backend.matmul(u[:, t, :], B)
        x_seq[:, t, :] = x_curr
        y_seq[:, t, :] = backend.matmul(x_curr, C)
        x_prev = x_curr

    return y_seq, x_seq


def s4_scan(u, A, B, C):
    """
    Simulated S4 sequential scan forward pass.
    y_t = x_t @ C,  x_t = x_{t-1} @ A + u_t @ B
    """
    y_seq, _ = _s4_forward_cache(u, A, B, C)
    return y_seq


@vjp_rule(func=s4_scan)
def _vjp_s4_scan(g, u, A, B, C):
    """
    S4 VJP returning exact gradients for (u, A, B, C).
    Propagates the adjoint state backward through the linear dynamical system.
    """
    y_seq, x_seq = _s4_forward_cache(u, A, B, C)
    batch, seq_len, d_in = u.shape
    d_model = A.shape[-1]

    gx = backend.zeros((batch, d_model), dtype=g.dtype)
    du = backend.zeros_like(u)
    dA = backend.zeros_like(A)
    dB = backend.zeros_like(B)
    dC = backend.zeros_like(C)

    # Completely vectorized batched matmuls inside the adjoint reverse sweep
    for t in range(seq_len - 1, -1, -1):
        g_y_t = g[:, t, :]
        x_t = x_seq[:, t, :]
        x_prev = x_seq[:, t - 1, :] if t > 0 else backend.zeros_like(x_t)

        dC += backend.matmul(x_t.T, g_y_t)

        gx = gx + backend.matmul(g_y_t, C.T)
        du[:, t, :] = backend.matmul(gx, B.T)
        dB += backend.matmul(u[:, t, :].T, gx)
        dA += backend.matmul(x_prev.T, gx)
        gx = backend.matmul(gx, A.T)

    return (du, dA, dB, dC)


# ---------------------------------------------------------------------------
# 8. Fast Fourier Transforms (FFT, IFFT, FFTN, IFFTN)
# ---------------------------------------------------------------------------

import math


def _fft_norm_scale(norm, n, is_forward):
    if norm is None or norm == "backward":
        return n if is_forward else (1.0 / n)
    elif norm == "ortho":
        return 1.0
    elif norm == "forward":
        return (1.0 / n) if is_forward else n
    raise ValueError(f"Unknown norm: {norm}")


def _slice_to_original(gx, orig_shape, axes):
    # If the padded output is strictly larger than the input, slice the gradient back down
    slices = [slice(None)] * gx.ndim
    for ax in axes:
        slices[ax] = slice(0, orig_shape[ax])
    return gx[tuple(slices)]


@vjp_rule(func=backend.scipy.fft.fft)
def _vjp_fft(g, x, n=None, axis=-1, norm=None):
    out_n = x.shape[axis] if n is None else n
    scale = _fft_norm_scale(norm, out_n, is_forward=True)
    gx = scale * backend.scipy.fft.ifft(g, n=out_n, axis=axis, norm=norm)
    if n is not None and n > x.shape[axis]:
        gx = _slice_to_original(gx, x.shape, [axis])
    return (gx,)


@vjp_rule(func=backend.scipy.fft.ifft)
def _vjp_ifft(g, x, n=None, axis=-1, norm=None):
    out_n = x.shape[axis] if n is None else n
    scale = _fft_norm_scale(norm, out_n, is_forward=False)
    gx = scale * backend.scipy.fft.fft(g, n=out_n, axis=axis, norm=norm)
    if n is not None and n > x.shape[axis]:
        gx = _slice_to_original(gx, x.shape, [axis])
    return (gx,)


@vjp_rule(func=backend.scipy.fft.fftn)
def _vjp_fftn(g, x, s=None, axes=None, norm=None):
    if axes is None:
        axes = tuple(range(x.ndim))
    if s is None:
        s = [x.shape[a] for a in axes]
    out_n = int(math.prod(s))
    scale = _fft_norm_scale(norm, out_n, is_forward=True)
    gx = scale * backend.scipy.fft.ifftn(g, s=s, axes=axes, norm=norm)
    if any(sz > x.shape[ax] for sz, ax in zip(s, axes)):
        gx = _slice_to_original(gx, x.shape, axes)
    return (gx,)


@vjp_rule(func=backend.scipy.fft.ifftn)
def _vjp_ifftn(g, x, s=None, axes=None, norm=None):
    if axes is None:
        axes = tuple(range(x.ndim))
    if s is None:
        s = [x.shape[a] for a in axes]
    out_n = int(math.prod(s))
    scale = _fft_norm_scale(norm, out_n, is_forward=False)
    gx = scale * backend.scipy.fft.fftn(g, s=s, axes=axes, norm=norm)
    if any(sz > x.shape[ax] for sz, ax in zip(s, axes)):
        gx = _slice_to_original(gx, x.shape, axes)
    return (gx,)
