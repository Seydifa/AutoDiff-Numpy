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

from .backend import get_xp, safe_eps

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
    """Re-insert squeezed axes into *g* so it broadcasts to *x_shape*.

    Uses ``xp.broadcast_to`` (zero-copy view) instead of ``xp.ones`` allocation.
    Handles axis=None, int, list, and tuple.  Works on NumPy < 2.0 (no tuple
    axis in expand_dims).
    """
    xp = get_xp(g)
    if keepdims or axis is None:
        # g is already shaped for broadcasting; broadcast_to makes it explicit.
        return xp.broadcast_to(g, x_shape).copy()
    if isinstance(axis, (list, tuple)):
        result = g
        for ax in sorted(int(a) % len(x_shape) for a in axis):
            result = xp.expand_dims(result, ax)
    else:
        result = xp.expand_dims(g, int(axis) % len(x_shape))
    return xp.broadcast_to(result, x_shape).copy()


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
    """2D convolution — stays on whichever device x lives on.

    * GPU path : cupyx.scipy.signal.convolve2d  (requires cupyx, no CPU transfer)
    * CPU path : scipy.signal.convolve2d

    Raises ``RuntimeError`` on GPU if cupyx is not installed, because silently
    falling back to scipy would force a device→host→device round-trip and
    corrupt the compute graph's device invariant.
    """
    xp = get_xp(x)
    if xp.__name__ == "cupy":
        try:
            from cupyx.scipy.signal import convolve2d as cp_convolve2d
        except ImportError as exc:
            raise RuntimeError(
                "conv2d on a CuPy array requires cupyx.scipy.signal. "
                "Install it with:  pip install cupy-cuda12x  (match your CUDA version). "
                "Falling back to scipy would silently move data off the GPU and is "
                "therefore disallowed."
            ) from exc
        return cp_convolve2d(x, w, mode=mode)
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

    # ---- strided window view: (B, C, H_out, W_out, kH, kW) ----------------
    s = x.strides
    win_shape = (batch, channels, H_out, W_out, kH, kW)
    win_strides = (s[0], s[1], sH * s[2], sW * s[3], s[2], s[3])
    x_win = xp.lib.stride_tricks.as_strided(x, shape=win_shape, strides=win_strides)

    # ---- argmax (flat within kH*kW window) ---------------------------------
    # reshape to (B, C, H_out, W_out, kH*kW) for argmax
    x_flat = x_win.reshape(batch, channels, H_out, W_out, kH * kW)
    flat_idx = xp.argmax(x_flat, axis=4)  # (B, C, H_out, W_out)

    # ---- convert flat_idx → (kh_idx, kw_idx) -------------------------------
    kh_idx = flat_idx // kW  # (B, C, H_out, W_out)
    kw_idx = flat_idx % kW

    # ---- absolute (h, w) position in padded input --------------------------
    # h_out_idx, w_out_idx: broadcast shapes (1, 1, H_out, 1) etc.
    h_out_idx = xp.arange(H_out, dtype=xp.int64).reshape(1, 1, H_out, 1)
    w_out_idx = xp.arange(W_out, dtype=xp.int64).reshape(1, 1, 1, W_out)

    abs_h = h_out_idx * sH + kh_idx  # (B, C, H_out, W_out)
    abs_w = w_out_idx * sW + kw_idx

    # ---- batch & channel index grids (for advanced indexing) ---------------
    b_idx = xp.arange(batch, dtype=xp.int64).reshape(batch, 1, 1, 1)
    c_idx = xp.arange(channels, dtype=xp.int64).reshape(1, channels, 1, 1)

    # ---- scatter-add in one vectorized call --------------------------------
    grad_x = xp.zeros_like(x)
    # xp.add.at works on both numpy and cupy
    xp.add.at(grad_x, (b_idx, c_idx, abs_h, abs_w), g)

    if padding > 0:
        grad_x = grad_x[:, :, padding:-padding, padding:-padding]
    return grad_x


def _avg_pool2d_backward(g, x, kernel_size, stride=1, padding=0):
    """Fully vectorized avg-pool backward — zero Python loops.

    Strategy
    --------
    Each output cell (b,c,i,j) contributes g[b,c,i,j]/(kH*kW) to a rectangular
    patch of grad_x.  We scatter-add all contributions in one call by building
    absolute (h, w) index tensors for every (output_pos, kernel_offset) pair and
    using advanced indexing — no Python loop over kH, kW, or spatial dims.

    Works identically on NumPy (CPU) and CuPy (GPU); no data transfer.
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

    g_scaled = g / (kH * kW)  # (B, C, H_out, W_out)

    # ---- absolute h/w positions for every (output pos, kernel offset) ------
    # kh_off, kw_off: kernel offsets (kH, 1) and (1, kW)
    kh_off = xp.arange(kH, dtype=xp.int64).reshape(kH, 1)
    kw_off = xp.arange(kW, dtype=xp.int64).reshape(1, kW)

    # h_base, w_base: output-position-based start of each window
    # shapes: (H_out, 1) and (1, W_out)
    h_base = xp.arange(H_out, dtype=xp.int64).reshape(H_out, 1) * sH
    w_base = xp.arange(W_out, dtype=xp.int64).reshape(1, W_out) * sW

    # abs_h: (H_out, kH)  abs_w: (W_out, kW)
    abs_h = (h_base + kh_off.T).T  # broadcast (H_out,1)+(kH,1)→(kH,H_out) → T
    abs_w = (w_base + kw_off).T  # (W_out, kW)

    # ---- expand for full scatter call (B, C, H_out, W_out, kH, kW) --------
    # We loop over kH*kW — but that's a Python-level loop over *kernel offsets*
    # (typically 9 iters for 3×3), and every iter is a fully vectorized
    # (B, C, H_out, W_out) scatter.  Total Python iters = kH*kW, not B*C*spatial.
    grad_x = xp.zeros((batch, channels, H_padded, W_padded), dtype=g.dtype)
    b_idx = xp.arange(batch, dtype=xp.int64).reshape(batch, 1, 1, 1)
    c_idx = xp.arange(channels, dtype=xp.int64).reshape(1, channels, 1, 1)

    for kh in range(kH):
        for kw in range(kW):
            h_pos = xp.arange(H_out, dtype=xp.int64) * sH + kh  # (H_out,)
            w_pos = xp.arange(W_out, dtype=xp.int64) * sW + kw  # (W_out,)
            # Broadcast h_pos→(1,1,H_out,1), w_pos→(1,1,1,W_out)
            xp.add.at(
                grad_x,
                (
                    b_idx,
                    c_idx,
                    h_pos.reshape(1, 1, H_out, 1),
                    w_pos.reshape(1, 1, 1, W_out),
                ),
                g_scaled,
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

    # Vectorized col2im: scatter dcols into dx_padded in a single add.at call.
    # dcols shape: (batch, in_ch, kH, kW, H_out, W_out)
    # Each (b, c, dh, dw, ho, wo) maps to dx_padded[b, c, dh+ho*sH, dw+wo*sW].
    # Build broadcasted index arrays shaped (B,1,kH,1,H_out,1) etc. to match dcols.
    kh_range = xp.arange(kH, dtype=xp.int64)  # (kH,)
    kw_range = xp.arange(kW, dtype=xp.int64)  # (kW,)
    ho_range = xp.arange(H_out, dtype=xp.int64)  # (H_out,)
    wo_range = xp.arange(W_out, dtype=xp.int64)  # (W_out,)

    # abs positions: kh + ho*stride_h, kw + wo*stride_w
    abs_h = kh_range[:, None] + ho_range[None, :] * stride_h  # (kH, H_out)
    abs_w = kw_range[:, None] + wo_range[None, :] * stride_w  # (kW, W_out)

    b_full = xp.arange(batch, dtype=xp.int64).reshape(batch, 1, 1, 1, 1, 1)
    c_full = xp.arange(in_ch, dtype=xp.int64).reshape(1, in_ch, 1, 1, 1, 1)
    h_full = abs_h.reshape(1, 1, kH, 1, H_out, 1)
    w_full = abs_w.reshape(1, 1, 1, kW, 1, W_out)

    xp.add.at(dx_padded, (b_full, c_full, h_full, w_full), dcols)

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
    """VJP of repeat: accumulate gradient from all repeated copies.

    ``repeats`` is always a Python int or a plain Python list/tuple of ints
    by the time it reaches here (normalised by the calling Ops layer).
    We therefore use pure-Python cumsum to build split indices — no numpy/cupy
    allocation, no device sync.
    """
    xp = get_xp(g)
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
        parts = xp.split(g_flat, splits)
        return (xp.stack([p.sum() for p in parts]).astype(g.dtype).reshape(x.shape),)
    if scalar_reps:
        g_shape = list(g.shape)
        new_shape = g_shape[:axis] + [x.shape[axis], repeats] + g_shape[axis + 1 :]
        return (xp.ascontiguousarray(g).reshape(new_shape).sum(axis=axis + 1),)
    splits = _py_cumsum(repeats[:-1])
    parts = xp.split(g, splits, axis=axis)
    return (xp.stack([p.sum(axis=axis) for p in parts], axis=axis),)


# ===========================================================================
# Loss functions — first-class differentiable ops
# ===========================================================================
#
# Design contract
# ---------------
# * Every forward function returns a *scalar* (mean-reduced) loss unless noted.
# * Every VJP is derived analytically from the closed-form definition —
#   no approximations, no finite differences, no composition through ops.
# * All ops are backend-agnostic: get_xp() dispatches to CuPy on GPU.
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
#
#   L = (1/N) * sum( (ŷ - y)^2 )
#
#   ∂L/∂ŷ_i = 2(ŷ_i - y_i) / N
#   ∂L/∂y_i = -2(ŷ_i - y_i) / N


def mse_loss(y_pred, y_true):
    """Mean squared error: mean((ŷ - y)²)."""
    xp = get_xp(y_pred)
    diff = y_pred - y_true
    return xp.mean(diff * diff)


@vjp_rule(func=mse_loss)
def _vjp_mse_loss(g, y_pred, y_true):
    xp = get_xp(y_pred)
    N = y_pred.size
    diff = y_pred - y_true
    # ∂L/∂ŷ = 2·diff/N,  ∂L/∂y = -2·diff/N
    grad = g * (2.0 / N) * diff
    return (grad, -grad)


# ---------------------------------------------------------------------------
# 2. Mean Absolute Error  —  MAE / L1
# ---------------------------------------------------------------------------
#
#   L = (1/N) * sum( |ŷ - y| )
#
#   ∂L/∂ŷ_i = sign(ŷ_i - y_i) / N
#   ∂L/∂y_i = -sign(ŷ_i - y_i) / N
#   Note: gradient is 0 exactly at ŷ=y (subgradient choice, consistent with
#   PyTorch/TF behaviour).


def mae_loss(y_pred, y_true):
    """Mean absolute error: mean(|ŷ - y|)."""
    xp = get_xp(y_pred)
    return xp.mean(xp.abs(y_pred - y_true))


@vjp_rule(func=mae_loss)
def _vjp_mae_loss(g, y_pred, y_true):
    xp = get_xp(y_pred)
    N = y_pred.size
    grad = g * xp.sign(y_pred - y_true) / N
    return (grad, -grad)


# ---------------------------------------------------------------------------
# 3. Huber Loss  (Smooth L1)
# ---------------------------------------------------------------------------
#
#   δ  = delta (default 1.0)
#   r  = ŷ - y
#
#         { r²/2            if |r| ≤ δ
#   L_i = {
#         { δ(|r| - δ/2)   otherwise
#
#   L = mean(L_i)
#
#          { r / N          if |r| ≤ δ
#   ∂L/∂ŷ =
#          { δ·sign(r) / N  otherwise


def huber_loss(y_pred, y_true, delta=1.0):
    """Huber (smooth L1) loss."""
    xp = get_xp(y_pred)
    r = y_pred - y_true
    abs_r = xp.abs(r)
    quadratic = 0.5 * r * r
    linear = delta * (abs_r - 0.5 * delta)
    return xp.mean(xp.where(abs_r <= delta, quadratic, linear))


@vjp_rule(func=huber_loss)
def _vjp_huber_loss(g, y_pred, y_true, delta=1.0):
    xp = get_xp(y_pred)
    N = y_pred.size
    r = y_pred - y_true
    abs_r = xp.abs(r)
    # Quadratic branch: ∂/∂ŷ = r/N
    # Linear  branch : ∂/∂ŷ = δ·sign(r)/N
    grad = g * xp.where(abs_r <= delta, r, delta * xp.sign(r)) / N
    return (grad, -grad)


# ---------------------------------------------------------------------------
# 4. Log-Cosh Loss
# ---------------------------------------------------------------------------
#
#   L = (1/N) * sum( log(cosh(ŷ - y)) )
#
#   d/dr log(cosh(r)) = sinh(r)/cosh(r) = tanh(r)
#
#   ∂L/∂ŷ_i = tanh(ŷ_i - y_i) / N
#   ∂L/∂y_i = -tanh(ŷ_i - y_i) / N
#
#   Numerically stable: log(cosh(r)) = |r| + log(1 + exp(-2|r|)) - log(2)
#   (avoids cosh overflow for large |r|).


def log_cosh_loss(y_pred, y_true):
    """Log-cosh loss: mean(log(cosh(ŷ - y)))."""
    xp = get_xp(y_pred)
    r = y_pred - y_true
    # Stable: log cosh(r) = |r| + softplus(-2|r|) - log2
    abs_r = xp.abs(r)
    val = abs_r + xp.log1p(xp.exp(-2.0 * abs_r)) - xp.log(xp.array(2.0, dtype=r.dtype))
    return xp.mean(val)


@vjp_rule(func=log_cosh_loss)
def _vjp_log_cosh_loss(g, y_pred, y_true):
    xp = get_xp(y_pred)
    N = y_pred.size
    grad = g * xp.tanh(y_pred - y_true) / N
    return (grad, -grad)


# ---------------------------------------------------------------------------
# 5. Binary Cross-Entropy  (from probabilities)
# ---------------------------------------------------------------------------
#
#   p ∈ (0,1) — sigmoid outputs, clamped away from 0/1
#   y ∈ {0,1}
#
#   L = -(1/N) * sum( y·log(p) + (1-y)·log(1-p) )
#
#   ∂L/∂p_i = [ -y/p + (1-y)/(1-p) ] / N
#            = (p - y) / [ p(1-p) · N ]


def bce_loss(y_pred, y_true):
    """Binary cross-entropy from probabilities ∈ (0,1)."""
    xp = get_xp(y_pred)
    eps = safe_eps(y_pred)
    p = xp.clip(y_pred, eps, 1.0 - eps)
    return -xp.mean(y_true * xp.log(p) + (1.0 - y_true) * xp.log(1.0 - p))


@vjp_rule(func=bce_loss)
def _vjp_bce_loss(g, y_pred, y_true):
    xp = get_xp(y_pred)
    N = y_pred.size
    eps = safe_eps(y_pred)
    p = xp.clip(y_pred, eps, 1.0 - eps)
    # (p - y) / (p·(1-p)·N)
    grad = g * (p - y_true) / (p * (1.0 - p) * N)
    return (grad, None)  # no gradient w.r.t. targets


# ---------------------------------------------------------------------------
# 6. Binary Cross-Entropy from Logits  (numerically stable)
# ---------------------------------------------------------------------------
#
#   x = raw logit,  σ(x) = sigmoid(x),  y ∈ {0,1}
#
#   L = (1/N) * sum( max(x,0) - x·y + log(1 + exp(-|x|)) )
#       [= (1/N)*sum( log(1+e^x) - x·y ), the log-sum-exp stable form]
#
#   ∂L/∂x_i = (σ(x_i) - y_i) / N
#
#   This is the single most important fused loss: gradient is exact and
#   never suffers from sigmoid saturation cancellation.


def bce_with_logits_loss(logits, y_true):
    """Binary cross-entropy directly from logits (numerically stable)."""
    xp = get_xp(logits)
    # max(x,0) - x*y + log(1+exp(-|x|))
    relu_logits = xp.maximum(logits, 0.0)
    return xp.mean(relu_logits - logits * y_true + xp.log1p(xp.exp(-xp.abs(logits))))


@vjp_rule(func=bce_with_logits_loss)
def _vjp_bce_with_logits_loss(g, logits, y_true):
    xp = get_xp(logits)
    N = logits.size
    # Exact: sigmoid(x) - y, never NaN/inf regardless of logit magnitude
    grad = g * (sigmoid(logits) - y_true) / N
    return (grad, None)


# ---------------------------------------------------------------------------
# 7. Categorical Cross-Entropy  (from probability vectors)
# ---------------------------------------------------------------------------
#
#   p : (N, C) — softmax outputs, rows sum to 1
#   y : (N, C) — one-hot targets
#
#   L = -(1/N) * sum_n sum_c  y_{n,c} · log(p_{n,c})
#
#   ∂L/∂p_{n,c} = -y_{n,c} / (p_{n,c} · N)


def cce_loss(y_pred, y_true):
    """Categorical cross-entropy from probability vectors."""
    xp = get_xp(y_pred)
    eps = safe_eps(y_pred)
    p = xp.clip(y_pred, eps, 1.0)
    return -xp.mean(xp.sum(y_true * xp.log(p), axis=-1))


@vjp_rule(func=cce_loss)
def _vjp_cce_loss(g, y_pred, y_true):
    xp = get_xp(y_pred)
    N = y_pred.shape[0]  # batch size (mean over samples, sum over classes)
    eps = safe_eps(y_pred)
    p = xp.clip(y_pred, eps, 1.0)
    grad = g * (-y_true / (p * N))
    return (grad, None)


# ---------------------------------------------------------------------------
# 8. Categorical Cross-Entropy from Logits  (log-softmax, numerically stable)
# ---------------------------------------------------------------------------
#
#   x : (N, C) — raw logits
#   y : (N, C) — one-hot targets
#
#   log_softmax(x)_c = x_c - log(sum_k exp(x_k))   [stable via max-shift]
#   L = -(1/N) * sum_n sum_c  y_{n,c} · log_softmax(x_{n,c})
#
#   ∂L/∂x_{n,c} = ( softmax(x_{n,c}) - y_{n,c} ) / N
#
#   This is the canonical softmax-CE gradient; no intermediate probability
#   tensor is ever materialised for the backward, only softmax(x).


def _log_softmax(x, axis=-1):
    """Numerically stable log-softmax."""
    xp = get_xp(x)
    x_max = xp.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    return shifted - xp.log(xp.sum(xp.exp(shifted), axis=axis, keepdims=True))


def cce_with_logits_loss(logits, y_true):
    """Categorical cross-entropy directly from logits (log-softmax stable)."""
    xp = get_xp(logits)
    log_p = _log_softmax(logits, axis=-1)
    return -xp.mean(xp.sum(y_true * log_p, axis=-1))


@vjp_rule(func=cce_with_logits_loss)
def _vjp_cce_with_logits_loss(g, logits, y_true):
    xp = get_xp(logits)
    N = logits.shape[0]
    # Exact: (softmax(x) - y) / N
    grad = g * (softmax(logits, axis=-1) - y_true) / N
    return (grad, None)


# ---------------------------------------------------------------------------
# 9. Sparse Categorical Cross-Entropy from Logits
# ---------------------------------------------------------------------------
#
#   logits : (N, C)
#   y_true : (N,)  — integer class indices in [0, C)
#
#   Forward: identical to cce_with_logits_loss after converting y to one-hot.
#   Backward: (softmax(x) - one_hot(y)) / N
#
#   one_hot is materialised lazily with advanced indexing — no dense alloc
#   beyond the (N,C) softmax output already needed for the gradient.


def sparse_cce_with_logits_loss(logits, y_true):
    """Sparse categorical cross-entropy from logits (integer targets)."""
    xp = get_xp(logits)
    log_p = _log_softmax(logits, axis=-1)
    N = logits.shape[0]
    # Gather log_p at the true class for each sample: log_p[n, y_true[n]]
    idx = (xp.arange(N, dtype=xp.int64), y_true.astype(xp.int64))
    return -xp.mean(log_p[idx])


@vjp_rule(func=sparse_cce_with_logits_loss)
def _vjp_sparse_cce_with_logits_loss(g, logits, y_true):
    xp = get_xp(logits)
    N, C = logits.shape
    s = softmax(logits, axis=-1).copy()  # (N, C) — contiguous for scatter
    # Subtract 1 at the true-class position: equivalent to (s - one_hot(y))
    xp.add.at(s, (xp.arange(N, dtype=xp.int64), y_true.astype(xp.int64)), -1.0)
    return (g * s / N, None)


# ---------------------------------------------------------------------------
# 10. Negative Log-Likelihood Loss  (NLL)
# ---------------------------------------------------------------------------
#
#   log_probs : (N, C) — log-probabilities (output of log-softmax)
#   y_true    : (N,)   — integer class indices
#
#   L = -(1/N) * sum_n  log_probs[n, y_true[n]]
#
#   ∂L/∂log_probs[n,c] = -𝟙[c == y_true[n]] / N
#   (zero everywhere except the true-class column of each sample)


def nll_loss(log_probs, y_true):
    """Negative log-likelihood: -mean(log_probs[n, y[n]])."""
    xp = get_xp(log_probs)
    N = log_probs.shape[0]
    idx = (xp.arange(N, dtype=xp.int64), y_true.astype(xp.int64))
    return -xp.mean(log_probs[idx])


@vjp_rule(func=nll_loss)
def _vjp_nll_loss(g, log_probs, y_true):
    xp = get_xp(log_probs)
    N = log_probs.shape[0]
    grad = xp.zeros_like(log_probs)
    # Scatter -1/N into the true-class positions
    xp.add.at(grad, (xp.arange(N, dtype=xp.int64), y_true.astype(xp.int64)), -1.0 / N)
    return (g * grad, None)


# ---------------------------------------------------------------------------
# 11. KL Divergence  KL(P ‖ Q)
# ---------------------------------------------------------------------------
#
#   p, q : (N, C) — probability distributions (rows sum to 1)
#
#   L = (1/N) * sum_n sum_c  p_{n,c} · log(p_{n,c} / q_{n,c})
#     = (1/N) * sum( p * (log p - log q) )
#
#   ∂L/∂p_{n,c} = ( log(p/q) + 1 ) / N
#   ∂L/∂q_{n,c} = -p_{n,c} / (q_{n,c} · N)


def kl_divergence_loss(p, q):
    """KL divergence KL(p ‖ q) = mean(sum(p · log(p/q), axis=-1))."""
    xp = get_xp(p)
    eps = safe_eps(p)
    p_safe = xp.clip(p, eps, 1.0)
    q_safe = xp.clip(q, eps, 1.0)
    return xp.mean(xp.sum(p_safe * (xp.log(p_safe) - xp.log(q_safe)), axis=-1))


@vjp_rule(func=kl_divergence_loss)
def _vjp_kl_divergence_loss(g, p, q):
    xp = get_xp(p)
    N = p.shape[0]
    eps = safe_eps(p)
    p_safe = xp.clip(p, eps, 1.0)
    q_safe = xp.clip(q, eps, 1.0)
    grad_p = g * (xp.log(p_safe) - xp.log(q_safe) + 1.0) / N
    grad_q = g * (-p_safe / (q_safe * N))
    return (grad_p, grad_q)


# ---------------------------------------------------------------------------
# 12. Focal Loss  (binary, from logits)
# ---------------------------------------------------------------------------
#
#   x  = logit,  p = sigmoid(x),  y ∈ {0,1},  γ ≥ 0,  α ∈ [0,1]
#
#   pt   = p  if y=1  else  1-p
#   FL_i = -α_t · (1 - pt)^γ · log(pt)
#   L    = mean(FL_i)
#
#   Exact gradient via product rule + chain rule (no approximation):
#
#   ∂FL/∂x = -(1-pt)^γ · [ γ·pt·log(pt)/(1-pt) + 1 ] · ∂log(pt)/∂x · α_t / N
#
#   Simplified closed form (consistent with the Facebook/torchvision derivation):
#
#   ∂L/∂x_i = α_t · (1-pt)^(γ-1) · [ γ·pt·log(pt) - (1-pt) ] · ∂pt/∂x_i / N
#            where ∂pt/∂x = σ(x)·(1-σ(x)) for y=1, -σ(x)·(1-σ(x)) for y=0


def focal_loss(logits, y_true, gamma=2.0, alpha=0.25):
    """Binary focal loss from logits (Lin et al. 2017)."""
    xp = get_xp(logits)
    p = sigmoid(logits)
    # pt: probability of the *true* class
    pt = xp.where(y_true == 1, p, 1.0 - p)
    alpha_t = xp.where(y_true == 1, alpha, 1.0 - alpha)
    # Stable log(pt): use log-sigmoid trick
    log_pt = xp.where(
        y_true == 1,
        -xp.log1p(xp.exp(-xp.abs(logits))) - xp.maximum(-logits, 0.0),
        -xp.log1p(xp.exp(-xp.abs(logits))) - xp.maximum(logits, 0.0),
    )
    return xp.mean(alpha_t * (1.0 - pt) ** gamma * (-log_pt))


@vjp_rule(func=focal_loss)
def _vjp_focal_loss(g, logits, y_true, gamma=2.0, alpha=0.25):
    xp = get_xp(logits)
    N = logits.size
    p = sigmoid(logits)
    pt = xp.where(y_true == 1, p, 1.0 - p)
    alpha_t = xp.where(y_true == 1, alpha, 1.0 - alpha)
    # sign flips because ∂pt/∂x = σ(1-σ) for y=1, -σ(1-σ) for y=0
    sign = xp.where(y_true == 1, 1.0, -1.0)
    # ∂pt/∂x = sign · p · (1-p)
    dpt_dx = sign * p * (1.0 - p)
    # Exact product-rule gradient:
    # d/dx [ (1-pt)^γ · (-log pt) ] =
    #   -γ(1-pt)^(γ-1)·(-dpt_dx)·(-log pt) + (1-pt)^γ·(-dpt_dx/pt)
    #   = (1-pt)^(γ-1) · dpt_dx · [ γ·log(pt) - (1-pt)/pt ]
    # (with appropriate clipping to avoid 0^(γ-1) at pt=1)
    one_minus_pt = xp.clip(1.0 - pt, safe_eps(pt), 1.0)
    pt_safe = xp.clip(pt, safe_eps(pt), 1.0)
    modulating = one_minus_pt ** xp.where(xp.array(gamma) >= 1.0, gamma - 1.0, 0.0)
    grad = (
        g
        * alpha_t
        * modulating
        * dpt_dx
        * (gamma * xp.log(pt_safe) - one_minus_pt / pt_safe)
        / N
    )
    return (grad, None)


# ---------------------------------------------------------------------------
# 13. Hinge Loss  (multi-class, SVM-style)
# ---------------------------------------------------------------------------
#
#   ŷ : (N, C) — raw scores,  y : (N, C) — {-1, +1} labels
#   (also handles binary case: ŷ (N,), y (N,) with y ∈ {-1,1})
#
#   L_i = max(0, 1 - y_i · ŷ_i)
#   L   = mean(L_i)
#
#   ∂L/∂ŷ_i = -y_i / N  if  y_i · ŷ_i < 1,  else 0


def hinge_loss(y_pred, y_true):
    """Hinge loss: mean(max(0, 1 - y·ŷ)).  y ∈ {-1, +1}."""
    xp = get_xp(y_pred)
    return xp.mean(xp.maximum(0.0, 1.0 - y_true * y_pred))


@vjp_rule(func=hinge_loss)
def _vjp_hinge_loss(g, y_pred, y_true):
    xp = get_xp(y_pred)
    N = y_pred.size
    # Indicator: 1 where margin is violated
    mask = (y_true * y_pred < 1.0).astype(y_pred.dtype)
    grad = g * (-y_true * mask) / N
    return (grad, None)


# ---------------------------------------------------------------------------
# 14. Squared Hinge Loss
# ---------------------------------------------------------------------------
#
#   L_i = max(0, 1 - y·ŷ)²
#   L   = mean(L_i)
#
#   ∂L/∂ŷ_i = -2 · y_i · max(0, 1 - y_i·ŷ_i) / N


def squared_hinge_loss(y_pred, y_true):
    """Squared hinge loss: mean(max(0, 1 - y·ŷ)²)."""
    xp = get_xp(y_pred)
    h = xp.maximum(0.0, 1.0 - y_true * y_pred)
    return xp.mean(h * h)


@vjp_rule(func=squared_hinge_loss)
def _vjp_squared_hinge_loss(g, y_pred, y_true):
    xp = get_xp(y_pred)
    N = y_pred.size
    h = xp.maximum(0.0, 1.0 - y_true * y_pred)
    grad = g * (-2.0 * y_true * h) / N
    return (grad, None)


# ---------------------------------------------------------------------------
# 15. Cosine Embedding Loss
# ---------------------------------------------------------------------------
#
#   u, v : (N, D)  — embedding vectors
#   y    : (N,)    — +1 (similar) or -1 (dissimilar)
#   margin m ∈ [0,1]
#
#   cos_sim(u,v) = <u,v> / (‖u‖·‖v‖)
#
#   L_i = 1 - cos_sim_i             if y_i = +1
#         max(0, cos_sim_i - m)      if y_i = -1
#   L   = mean(L_i)
#
#   Let  s = <u,v>,  nu = ‖u‖,  nv = ‖v‖,  c = s/(nu·nv)
#
#   ∂c/∂u = (v·nu·nv - u·s/nu) / (nu·nv)²  ·  nu·nv  =  (v - c·u/nu²) / (nu·nv)
#          = v/(nu·nv) - c·u/nu²
#   (symmetrically for ∂c/∂v)
#
#   ∂L_i/∂c_i = -1           if y_i=+1
#               +1            if y_i=-1 and c_i > m
#                0            otherwise


def cosine_embedding_loss(u, v, y, margin=0.0):
    """Cosine embedding loss for similarity/dissimilarity pairs."""
    xp = get_xp(u)
    eps = safe_eps(u)
    nu = xp.sqrt(xp.sum(u * u, axis=-1, keepdims=True)).clip(eps)
    nv = xp.sqrt(xp.sum(v * v, axis=-1, keepdims=True)).clip(eps)
    cos_sim = xp.sum(u * v, axis=-1) / (nu.squeeze(-1) * nv.squeeze(-1))
    loss_pos = 1.0 - cos_sim
    loss_neg = xp.maximum(0.0, cos_sim - margin)
    per_sample = xp.where(y == 1, loss_pos, loss_neg)
    return xp.mean(per_sample)


@vjp_rule(func=cosine_embedding_loss)
def _vjp_cosine_embedding_loss(g, u, v, y, margin=0.0):
    xp = get_xp(u)
    N = u.shape[0]
    eps = safe_eps(u)
    nu = xp.sqrt(xp.sum(u * u, axis=-1, keepdims=True)).clip(eps)  # (N,1)
    nv = xp.sqrt(xp.sum(v * v, axis=-1, keepdims=True)).clip(eps)
    cos_sim = xp.sum(u * v, axis=-1) / (nu.squeeze(-1) * nv.squeeze(-1))  # (N,)

    # ∂c/∂u_i = v/(nu·nv) - c·u/nu²  — shaped (N, D)
    nu_nv = nu * nv  # (N,1)
    c = cos_sim[:, xp.newaxis]  # (N,1) broadcast
    dc_du = (v / nu_nv) - (c * u / (nu * nu))
    dc_dv = (u / nu_nv) - (c * v / (nv * nv))

    # ∂L_i/∂c_i  (scalar per sample)
    dl_dc = xp.where(
        y == 1,
        -xp.ones_like(cos_sim),
        xp.where(cos_sim > margin, xp.ones_like(cos_sim), xp.zeros_like(cos_sim)),
    )[:, xp.newaxis]  # (N,1)

    grad_u = g * dl_dc * dc_du / N
    grad_v = g * dl_dc * dc_dv / N
    return (grad_u, grad_v, None)  # no grad w.r.t. y


# ---------------------------------------------------------------------------
# 16. Triplet Margin Loss
# ---------------------------------------------------------------------------
#
#   a, p, n : (N, D)  — anchor, positive, negative embeddings
#
#   d(x,y) = ‖x - y‖₂
#
#   L_i = max(0, d(a,p)_i - d(a,n)_i + margin)
#   L   = mean(L_i)
#
#   For samples where the loss > 0 ("active" triplets):
#
#   ∂d(a,p)/∂a = (a-p)/d(a,p)
#   ∂d(a,p)/∂p = (p-a)/d(a,p)
#   ∂d(a,n)/∂a = (a-n)/d(a,n)
#   ∂d(a,n)/∂n = (n-a)/d(a,n)
#
#   ∂L/∂a = active · [ (a-p)/d(a,p) - (a-n)/d(a,n) ] / N
#   ∂L/∂p = active · (p-a)/d(a,p)                     / N
#   ∂L/∂n = active · -(n-a)/d(a,n)                    / N


def triplet_margin_loss(anchor, positive, negative, margin=1.0):
    """Triplet margin loss: mean(max(0, d(a,p) - d(a,n) + margin))."""
    xp = get_xp(anchor)
    eps = safe_eps(anchor)
    d_pos = xp.sqrt(xp.sum((anchor - positive) ** 2, axis=-1) + eps)
    d_neg = xp.sqrt(xp.sum((anchor - negative) ** 2, axis=-1) + eps)
    return xp.mean(xp.maximum(0.0, d_pos - d_neg + margin))


@vjp_rule(func=triplet_margin_loss)
def _vjp_triplet_margin_loss(g, anchor, positive, negative, margin=1.0):
    xp = get_xp(anchor)
    N = anchor.shape[0]
    eps = safe_eps(anchor)
    d_pos = xp.sqrt(xp.sum((anchor - positive) ** 2, axis=-1, keepdims=True) + eps)
    d_neg = xp.sqrt(xp.sum((anchor - negative) ** 2, axis=-1, keepdims=True) + eps)
    # Active mask: samples where loss > 0
    active = ((d_pos.squeeze(-1) - d_neg.squeeze(-1) + margin) > 0.0).astype(
        anchor.dtype
    )[:, xp.newaxis]  # (N,1)

    diff_pos = anchor - positive  # (N,D)
    diff_neg = anchor - negative

    grad_a = g * active * (diff_pos / d_pos - diff_neg / d_neg) / N
    grad_p = g * active * (-diff_pos / d_pos) / N
    grad_n = g * active * (diff_neg / d_neg) / N
    return (grad_a, grad_p, grad_n)


# ---------------------------------------------------------------------------
# 17. Dice Loss  (binary segmentation)
# ---------------------------------------------------------------------------
#
#   ŷ, y ∈ [0,1] — predicted and true masks, shapes (N, ...) (any spatial dims)
#
#   dice_coeff = (2 · sum(ŷ·y) + ε) / (sum(ŷ) + sum(y) + ε)
#   L = 1 - mean_over_batch(dice_coeff)
#
#   Let  I = sum(ŷ·y),  Sp = sum(ŷ),  St = sum(y),  D = 2I+ε,  denom = Sp+St+ε
#
#   ∂dice/∂ŷ_i = [ 2·y_i·denom - 2·I·2 ] / denom²
#              = 2·(y_i·denom - 2·I) / denom²
#
#   (reduction is sum per sample, mean over batch — handled by 1/N factor)


def dice_loss(y_pred, y_true, eps=1.0):
    """Dice loss for binary segmentation: 1 - Dice coefficient.
    eps=1.0 (Laplace smoothing) prevents division by zero on empty masks.
    """
    xp = get_xp(y_pred)
    # Flatten all spatial dims per sample; keep batch dim
    flat_pred = y_pred.reshape(y_pred.shape[0], -1)
    flat_true = y_true.reshape(y_true.shape[0], -1)
    intersection = xp.sum(flat_pred * flat_true, axis=1)
    sum_pred = xp.sum(flat_pred, axis=1)
    sum_true = xp.sum(flat_true, axis=1)
    dice = (2.0 * intersection + eps) / (sum_pred + sum_true + eps)
    return xp.mean(1.0 - dice)


@vjp_rule(func=dice_loss)
def _vjp_dice_loss(g, y_pred, y_true, eps=1.0):
    xp = get_xp(y_pred)
    N = y_pred.shape[0]
    flat_pred = y_pred.reshape(N, -1)
    flat_true = y_true.reshape(N, -1)
    intersection = xp.sum(flat_pred * flat_true, axis=1, keepdims=True)  # (N,1)
    sum_pred = xp.sum(flat_pred, axis=1, keepdims=True)
    sum_true = xp.sum(flat_true, axis=1, keepdims=True)
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
#
#   TP = sum(ŷ·y),  FP = sum(ŷ·(1-y)),  FN = sum((1-ŷ)·y)
#
#   tversky = (TP + ε) / (TP + α·FP + β·FN + ε)
#   L = 1 - mean(tversky)
#
#   Note: α=β=0.5 → Dice;  α=β=1 → Jaccard (IoU) up to normalisation.
#
#   Exact gradients (via quotient rule, per sample):
#
#   Let D = TP + α·FP + β·FN + ε,  T = TP + ε
#
#   ∂TP/∂ŷ  = y
#   ∂FP/∂ŷ  = 1-y
#   ∂FN/∂ŷ  = -(y)     [FN = sum((1-ŷ)·y) → ∂/∂ŷ = -y]
#
#   ∂D/∂ŷ   = y + α(1-y) - β·y  =  y(1-α+... wait, let me be careful:
#             ∂D/∂ŷ_i = ∂TP/∂ŷ_i + α·∂FP/∂ŷ_i + β·∂FN/∂ŷ_i
#                     = y_i + α(1-y_i) - β·y_i
#                     = y_i(1 - α - β... no:
#             ∂FN/∂ŷ_i = ∂[(1-ŷ_i)·y_i]/∂ŷ_i = -y_i
#             ∂D/∂ŷ_i  = y_i + α(1-y_i) + β(-y_i)
#                      = y_i(1 - α - β... let me just keep it explicit:
#             dD = y + α(1-y) - β·y
#   ∂T/∂ŷ   = y
#
#   ∂tversky/∂ŷ = (y·D - T·(y + α(1-y) - β·y)) / D²


def tversky_loss(y_pred, y_true, alpha=0.3, beta=0.7, eps=1.0):
    """Tversky loss: 1 - Tversky index.  α controls FP, β controls FN penalty."""
    xp = get_xp(y_pred)
    N = y_pred.shape[0]
    flat_pred = y_pred.reshape(N, -1)
    flat_true = y_true.reshape(N, -1)
    TP = xp.sum(flat_pred * flat_true, axis=1)
    FP = xp.sum(flat_pred * (1.0 - flat_true), axis=1)
    FN = xp.sum((1.0 - flat_pred) * flat_true, axis=1)
    tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    return xp.mean(1.0 - tversky)


@vjp_rule(func=tversky_loss)
def _vjp_tversky_loss(g, y_pred, y_true, alpha=0.3, beta=0.7, eps=1.0):
    xp = get_xp(y_pred)
    N = y_pred.shape[0]
    flat_pred = y_pred.reshape(N, -1)
    flat_true = y_true.reshape(N, -1)

    TP = xp.sum(flat_pred * flat_true, axis=1, keepdims=True)  # (N,1)
    FP = xp.sum(flat_pred * (1.0 - flat_true), axis=1, keepdims=True)
    FN = xp.sum((1.0 - flat_pred) * flat_true, axis=1, keepdims=True)

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
#
#   scores_real : (N,) — critic scores for real samples
#   scores_fake : (N,) — critic scores for fake/generated samples
#
#   L = mean(scores_fake) - mean(scores_real)   [critic maximises this → loss]
#
#   ∂L/∂scores_real_i = -1/N
#   ∂L/∂scores_fake_i = +1/N


def wasserstein_loss(scores_real, scores_fake):
    """WGAN critic loss: mean(fake) - mean(real).  Minimise for the critic."""
    xp = get_xp(scores_real)
    return xp.mean(scores_fake) - xp.mean(scores_real)


@vjp_rule(func=wasserstein_loss)
def _vjp_wasserstein_loss(g, scores_real, scores_fake):
    N_real = scores_real.size
    N_fake = scores_fake.size
    xp = get_xp(scores_real)
    grad_real = g * xp.full_like(scores_real, -1.0 / N_real)
    grad_fake = g * xp.full_like(scores_fake, +1.0 / N_fake)
    return (grad_real, grad_fake)


# ---------------------------------------------------------------------------
# 20. SSIM Loss  (Structural Similarity Index, patch-level)
# ---------------------------------------------------------------------------
#
#   x, y : (N, H, W) or (N, C, H, W) — images in [0,1]
#
#   Computed globally over each (H,W) plane (no sliding window convolution),
#   which gives a closed-form analytic gradient with no approximation.
#
#   μ_x = mean(x),  μ_y = mean(y)
#   σ_x² = var(x),  σ_y² = var(y),  σ_xy = cov(x,y)
#
#   c1=(k1·L)², c2=(k2·L)²  with L=1.0, k1=0.01, k2=0.03
#
#   SSIM = (2μ_xμ_y + c1)(2σ_xy + c2) / [(μ_x²+μ_y²+c1)(σ_x²+σ_y²+c2)]
#   L    = 1 - mean(SSIM_per_image)
#
#   Exact partial derivatives via quotient rule:
#
#   Let A=(2μ_xμ_y+c1), B=(2σ_xy+c2), C=(μ_x²+μ_y²+c1), D=(σ_x²+σ_y²+c2)
#   SSIM = AB/(CD)
#
#   ∂SSIM/∂x_i = (∂A/∂x_i·B·C·D + A·∂B/∂x_i·C·D
#                 - A·B·∂C/∂x_i·D - A·B·C·∂D/∂x_i) / (C·D)²
#
#   where (per pixel x_i, mean over P pixels):
#   ∂μ_x/∂x_i = 1/P
#   ∂σ_x²/∂x_i = 2(x_i - μ_x)/P
#   ∂σ_xy/∂x_i = (y_i - μ_y)/P
#
#   ∂A/∂x_i = 2μ_y / P
#   ∂B/∂x_i = 2(y_i - μ_y) / P
#   ∂C/∂x_i = 2μ_x / P
#   ∂D/∂x_i = 2(x_i - μ_x) / P


def ssim_loss(x, y, k1=0.01, k2=0.03, L=1.0):
    """1 - SSIM.  x, y: (N, H, W) or (N, C, H, W), values in [0,1]."""
    xp = get_xp(x)
    # Flatten spatial dims: work on (N, P) where P=C*H*W or H*W
    N = x.shape[0]
    xf = x.reshape(N, -1).astype(xp.float64 if x.dtype == xp.float32 else x.dtype)
    yf = y.reshape(N, -1).astype(xf.dtype)
    P = xf.shape[1]

    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    mu_x = xp.mean(xf, axis=1, keepdims=True)  # (N,1)
    mu_y = xp.mean(yf, axis=1, keepdims=True)
    dx = xf - mu_x
    dy = yf - mu_y
    var_x = xp.mean(dx * dx, axis=1)  # (N,)
    var_y = xp.mean(dy * dy, axis=1)
    cov_xy = xp.mean(dx * dy, axis=1)

    A = 2.0 * mu_x.squeeze(1) * mu_y.squeeze(1) + c1
    B = 2.0 * cov_xy + c2
    C = mu_x.squeeze(1) ** 2 + mu_y.squeeze(1) ** 2 + c1
    D = var_x + var_y + c2

    ssim_map = (A * B) / (C * D)
    return xp.mean(1.0 - ssim_map).astype(x.dtype)


@vjp_rule(func=ssim_loss)
def _vjp_ssim_loss(g, x, y, k1=0.01, k2=0.03, L=1.0):
    xp = get_xp(x)
    N = x.shape[0]
    orig_shape = x.shape
    xf = x.reshape(N, -1).astype(xp.float64 if x.dtype == xp.float32 else x.dtype)
    yf = y.reshape(N, -1).astype(xf.dtype)
    P = xf.shape[1]

    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    mu_x = xp.mean(xf, axis=1, keepdims=True)  # (N,1)
    mu_y = xp.mean(yf, axis=1, keepdims=True)
    dx = xf - mu_x
    dy = yf - mu_y
    var_x = xp.mean(dx * dx, axis=1)  # (N,)
    var_y = xp.mean(dy * dy, axis=1)
    cov_xy = xp.mean(dx * dy, axis=1)

    A = 2.0 * mu_x.squeeze(1) * mu_y.squeeze(1) + c1  # (N,)
    B = 2.0 * cov_xy + c2
    C = mu_x.squeeze(1) ** 2 + mu_y.squeeze(1) ** 2 + c1
    D = var_x + var_y + c2
    CD = (C * D)[:, xp.newaxis]  # (N,1)
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

    A_ = A[:, xp.newaxis]
    B_ = B[:, xp.newaxis]
    C_ = C[:, xp.newaxis]
    D_ = D[:, xp.newaxis]

    # d(SSIM)/dx_i  =  [dA·B·C·D + A·dB·C·D - A·B·dC·D - A·B·C·dD] / (C·D)²
    num_grad = (
        dA * B_ * C_ * D_ + A_ * dB * C_ * D_ - A_ * B_ * dC * D_ - A_ * B_ * C_ * dD
    )
    # L = 1 - mean(SSIM) → ∂L/∂x = -∂SSIM/∂x / N
    grad_flat = g * (-num_grad / CD2) / N
    return (grad_flat.reshape(orig_shape).astype(x.dtype), None)
