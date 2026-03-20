"""
dnp/core/vjp_rules.py
=====================
Vector-Jacobian Product (VJP) rules for all supported operations,
plus activation functions and convolution helpers.

This is the single source of truth — previously split across dnp/ops/vjp_rules.py.
"""

# Third-party libraries
import numpy as np

from .backend import get_xp, as_numpy

EPSILON = 1e-8

# Precomputed constant for GELU — avoids recomputing sqrt on every call.
_GELU_COEFF = float(np.sqrt(2.0 / np.pi))  # ≈ 0.7978845608028654

# ---------------------------------------------------------------------------
# Vectorized kernels (CPU & GPU compatible via get_xp)
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
    # No data is copied — this is a zero-cost view of x_padded.
    s = x_padded.strides
    col_shape = (batch, in_ch, kH, kW, H_out, W_out)
    col_strides = (s[0], s[1], s[2], s[3], stride_h * s[2], stride_w * s[3])
    cols = xp.lib.stride_tricks.as_strided(
        x_padded, shape=col_shape, strides=col_strides
    )

    # Reshape to 2-D for a single batched matmul.
    # ascontiguousarray ensures the overlapping view can be reshaped safely.
    cols_2d = xp.ascontiguousarray(cols).reshape(batch, in_ch * kH * kW, H_out * W_out)
    W_2d = W.reshape(out_ch, in_ch * kH * kW)
    return xp.matmul(W_2d, cols_2d).reshape(batch, out_ch, H_out, W_out)


# ---------------------------------------------------------------------------
# Broadcasting utility
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
    # logaddexp(0, x) = log(1 + exp(x)) computed in a numerically stable way;
    # avoids manual clipping, one fewer intermediate array.
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
# Pooling operations (vectorized fallback or cpu bound logic where necessary)
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
    """Dropout forward — stores the binary mask so the VJP can reuse it.

    Returns the masked/scaled output.  The mask is NOT stored inside this
    function; the mask is generated in ``_DropoutOps._raw_call`` which writes
    it into ``op_kwargs["mask"]`` so ``Tensor.backward()`` can retrieve it.
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
        # Fallback to scipy or cupyx if available
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
# Backward pass for the vectorised im2col convolution used by the Conv2d layer
# ---------------------------------------------------------------------------


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

    # 1. Reconstruct the padded input
    if pad_h > 0 or pad_w > 0:
        x_padded = xp.pad(x_unpad, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    else:
        x_padded = x_unpad

    # 2. Im2col: (batch, in_ch*kH*kW, H_out*W_out)
    s = x_padded.strides
    col_shape = (batch, in_ch, kH, kW, H_out, W_out)
    col_strides = (s[0], s[1], s[2], s[3], stride_h * s[2], stride_w * s[3])
    cols = xp.lib.stride_tricks.as_strided(
        x_padded, shape=col_shape, strides=col_strides
    )
    cols_2d = xp.ascontiguousarray(cols).reshape(
        batch, in_ch * kH * kW, H_out * W_out
    )  # (B, K, N)  where K=in_ch*kH*kW, N=H_out*W_out

    # 3. Reshape upstream gradient
    g_2d = g.reshape(batch, out_ch, H_out * W_out)  # (B, C_out, N)
    W_2d = W.reshape(out_ch, in_ch * kH * kW)  # (C_out, K)

    # 4. Gradient w.r.t. W
    # dW_2d[c, k] = sum_{b, n} g_2d[b, c, n] * cols_2d[b, k, n]
    # = (g_2d @ cols_2d^T).sum(batch)  shape: (C_out, K)
    dW_2d = xp.matmul(g_2d, cols_2d.transpose(0, 2, 1)).sum(axis=0)
    dW = dW_2d.reshape(W.shape)  # (C_out, C_in, kH, kW)

    # 5. Gradient w.r.t. cols_2d via W^T @ g
    # dcols_2d[b, k, n] = sum_{c} W_2d[c, k] * g_2d[b, c, n]
    # Vectorised: (1, K, C_out) @ (B, C_out, N) → (B, K, N)
    dcols_2d = xp.matmul(
        W_2d.T[None],  # broadcast over batch: (1, K, C_out)
        g_2d,  # (B, C_out, N)
    )  # → (B, K, N)

    # 6. col2im: scatter-add dcols back to dx_padded
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

    # 7. Strip padding to recover dx with same shape as x_unpad
    if pad_h > 0 or pad_w > 0:
        dx = dx_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
    else:
        dx = dx_padded

    return dx, dW


# ---------------------------------------------------------------------------
# VJP helpers — named functions that compute each activation once, then reuse
# the cached result for both the output value and the gradient multiplier.
# ---------------------------------------------------------------------------


def _vjp_sigmoid(g, x):
    s = sigmoid(x)
    return (g * s * (1.0 - s),)


def _vjp_swish(g, x):
    s = sigmoid(x)
    sw = x * s
    return (g * (sw + s * (1.0 - sw)),)


def _vjp_gelu(g, x):
    xp = get_xp(x)
    inner = _GELU_COEFF * (x + 0.044715 * x**3)
    t = xp.tanh(inner)  # computed once, reused twice
    dcdf = _GELU_COEFF * (1.0 + 3.0 * 0.044715 * x**2)
    return (g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * dcdf),)


def _vjp_softmax(g, x):
    xp = get_xp(x)
    s = softmax(x)  # computed once, reused twice
    return (s * (g - xp.sum(g * s, axis=-1, keepdims=True)),)


# ---------------------------------------------------------------------------
# Reduction-axis helper
# ---------------------------------------------------------------------------


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
# VJP Rules dictionary
# ---------------------------------------------------------------------------

VJP_RULES = {
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
        unbroadcast(g * y * get_xp(x).power(x, y - 1), x.shape),
        unbroadcast(g * get_xp(x).power(x, y) * get_xp(x).log(x + EPSILON), y.shape),
    ),
    np.maximum: lambda g, x, y: (
        unbroadcast(g * (x >= y), x.shape),
        unbroadcast(g * (x < y), y.shape),
    ),
    np.minimum: lambda g, x, y: (
        unbroadcast(g * (x <= y), x.shape),
        unbroadcast(g * (x > y), y.shape),
    ),
    # Unary Operations
    np.negative: lambda g, x: (-g,),
    np.square: lambda g, x: (g * 2 * x,),
    np.sqrt: lambda g, x: (g / (2 * get_xp(x).sqrt(x) + EPSILON),),
    np.exp: lambda g, x: (g * get_xp(x).exp(x),),
    np.log: lambda g, x: (g / (x + EPSILON),),
    np.log1p: lambda g, x: (g / (x + 1.0 + EPSILON),),
    np.expm1: lambda g, x: (g * get_xp(x).exp(x),),
    np.abs: lambda g, x: (g * get_xp(x).sign(x),),
    np.sign: lambda g, x: (get_xp(x).zeros_like(x),),
    # Trig
    np.sin: lambda g, x: (g * get_xp(x).cos(x),),
    np.cos: lambda g, x: (g * -get_xp(x).sin(x),),
    np.tan: lambda g, x: (g / (get_xp(x).cos(x) ** 2 + EPSILON),),
    np.sinh: lambda g, x: (g * get_xp(x).cosh(x),),
    np.cosh: lambda g, x: (g * get_xp(x).sinh(x),),
    np.tanh: lambda g, x: (g * (1 - get_xp(x).tanh(x) ** 2),),
    # Rounding
    np.floor: lambda g, x: (get_xp(x).zeros_like(x),),
    np.ceil: lambda g, x: (get_xp(x).zeros_like(x),),
    np.round: lambda g, x: (get_xp(x).zeros_like(x),),
    # Matrix operations
    np.matmul: lambda g, x, y: (
        unbroadcast(
            get_xp(g).matmul(
                g, get_xp(y).swapaxes(y, -1, -2) if getattr(y, "ndim", 0) >= 2 else y
            ),
            x.shape,
        ),
        unbroadcast(
            get_xp(g).matmul(
                get_xp(x).swapaxes(x, -1, -2) if getattr(x, "ndim", 0) >= 2 else x, g
            ),
            y.shape,
        ),
    ),
    np.dot: lambda g, x, y: (get_xp(g).dot(g, y.T), get_xp(g).dot(x.T, g)),
    # Reductions
    np.sum: lambda g, x, axis=None, keepdims=False: (
        _restore_reduced_dims(g, x.shape, axis, keepdims),
    ),
    np.mean: lambda g, x, axis=None, keepdims=False: (
        _restore_reduced_dims(g, x.shape, axis, keepdims)
        / (
            int(
                get_xp(x).prod(
                    get_xp(x).array(
                        [
                            x.shape[a]
                            for a in (
                                axis if isinstance(axis, (list, tuple)) else [axis]
                            )
                        ]
                    )
                )
            )
            if axis is not None
            else x.size
        ),
    ),
    np.prod: lambda g, x, axis=None, keepdims=False: (
        _restore_reduced_dims(g, x.shape, axis, keepdims)
        * (get_xp(x).prod(x, axis=axis, keepdims=True) / (x + EPSILON)),
    ),
    np.max: lambda g, x, axis=None, keepdims=False: (
        _restore_reduced_dims(g, x.shape, axis, keepdims)
        * (x == get_xp(x).max(x, axis=axis, keepdims=True)).astype(g.dtype),
    ),
    np.min: lambda g, x, axis=None, keepdims=False: (
        _restore_reduced_dims(g, x.shape, axis, keepdims)
        * (x == get_xp(x).min(x, axis=axis, keepdims=True)).astype(g.dtype),
    ),
    # Shape operations — _reshape is populated after the dict (see below).
    np.transpose: lambda g, x, axes=None: (
        # np.argsort is used intentionally — axes is always a tiny Python list/tuple,
        # and cp.argsort(tuple) fails on some CuPy versions.  The result is passed
        # back as a plain list so both NumPy and CuPy accept it for transpose.
        get_xp(g).transpose(g, np.argsort(list(axes)).tolist())
        if axes is not None
        else g.T,
    ),
    np.expand_dims: lambda g, x, axis: (get_xp(g).squeeze(g, axis),),
    np.squeeze: lambda g, x, axis=None: (
        get_xp(g).expand_dims(g, axis)
        if axis is not None
        else get_xp(g).reshape(g, x.shape),
    ),
    # NN ops
    conv2d: lambda g, x, w, mode="valid": (
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
    ),
    sigmoid: _vjp_sigmoid,
    relu: lambda g, x: (g * (x > 0).astype(x.dtype),),
    leaky_relu: lambda g, x: (g * get_xp(x).where(x > 0, 1.0, 0.01),),
    elu: lambda g, x: (g * get_xp(x).where(x > 0, 1.0, 1.0 * get_xp(x).exp(x)),),
    softplus: lambda g, x: (g * sigmoid(x),),
    swish: _vjp_swish,
    gelu: _vjp_gelu,
    softmax: _vjp_softmax,
    max_pool2d: lambda g, x, kernel_size, stride=1, padding=0: (
        _max_pool2d_backward(g, x, kernel_size, stride, padding),
    ),
    avg_pool2d: lambda g, x, kernel_size, stride=1, padding=0: (
        _avg_pool2d_backward(g, x, kernel_size, stride, padding),
    ),
    dropout: lambda g, x, p=0.5, training=True, mask=None: (
        # Use the recorded binary mask; fall back gracefully if absent.
        (g * mask,)
        if (training and mask is not None)
        else (g / (1 - p),)
        if training
        else (g,)
    ),
    batch_norm: lambda g, x, weight, bias, mean, var, eps=1e-5: _batch_norm_backward(
        g, x, weight, bias, mean, var, eps
    ),
}

# Register the im2col conv VJP after the dict so that _conv2d_forward_kernel
# (defined at the top of this module) can be used as the dict key.
VJP_RULES[_conv2d_forward_kernel] = _vjp_conv2d_forward_kernel

# ---------------------------------------------------------------------------
# _reshape: NumPy-version-safe wrapper for np.reshape
# ---------------------------------------------------------------------------
# NumPy 2.0 renamed the `newshape` keyword argument to `shape`.  Passing the
# new shape *positionally* works on every NumPy version, so we wrap it here.


def _reshape(a, newshape):
    """NumPy-version-agnostic reshape (passes newshape positionally)."""
    return np.reshape(a, newshape)


# Register VJP rule after both the dict AND _reshape are defined.
VJP_RULES[_reshape] = lambda g, x, newshape: (get_xp(g).reshape(g, x.shape),)
# Keep the np.reshape key as an alias so existing code that looks up
# VJP_RULES[np.reshape] directly still works.
VJP_RULES[np.reshape] = VJP_RULES[_reshape]


# ---------------------------------------------------------------------------
# Backward pass helpers
# ---------------------------------------------------------------------------


def _max_pool2d_backward(g, x, kernel_size, stride=1, padding=0):
    """Vectorized max-pool backward via strided mask + kH*kW scatter-adds.

    Each of the kH*kW iterations is a fully vectorized operation over
    (batch, channels, H_out, W_out) — no Python loops over spatial dims.
    Works on both NumPy (CPU) and CuPy (GPU) without any data transfer.
    """
    xp = get_xp(x)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    kH, kW = kernel_size
    sH, sW = stride

    if padding > 0:
        x = xp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    batch, channels, H_pad, W_pad = x.shape
    H_out = (H_pad - kH) // sH + 1
    W_out = (W_pad - kW) // sW + 1

    # Zero-copy strided view: (B, C, H_out, W_out, kH, kW)
    s = x.strides
    win_shape = (batch, channels, H_out, W_out, kH, kW)
    win_strides = (s[0], s[1], sH * s[2], sW * s[3], s[2], s[3])
    x_win = xp.lib.stride_tricks.as_strided(x, shape=win_shape, strides=win_strides)

    # Boolean mask where each window attains its maximum; ties split equally.
    max_val = xp.max(x_win, axis=(4, 5), keepdims=True)
    mask = (x_win == max_val).astype(g.dtype)
    mask /= mask.sum(axis=(4, 5), keepdims=True)

    # Weighted upstream gradient broadcast over kernel dims.
    grad_win = g[:, :, :, :, xp.newaxis, xp.newaxis] * mask  # (B,C,H_out,W_out,kH,kW)

    # Scatter-add: kH*kW vectorized writes, indexed by kernel offset.
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
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    kH, kW = kernel_size
    sH, sW = stride

    batch, channels, H, W = x.shape
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    H_out = (H_padded - kH) // sH + 1
    W_out = (W_padded - kW) // sW + 1

    g_scaled = g / (kH * kW)  # uniform contribution per input element
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

    # Determine the reduction axes: for BatchNorm2d, normalization is over
    # (batch, H, W) simultaneously (axes 0, 2, 3); for BatchNorm1d, only
    # axis 0.  We infer this from the rank of the input.
    if x.ndim == 4:
        # BatchNorm2d: x shape = (B, C, H, W), mean/var shape = (1, C, 1, 1)
        norm_axes = (0, 2, 3)
        N = x.shape[0] * x.shape[2] * x.shape[3]
    else:
        # BatchNorm1d: x shape = (B, C), mean/var shape = (C,)
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
    # Gradients for the affine parameters γ (weight) and β (bias)
    grad_weight = unbroadcast(g * x_norm, weight.shape)
    grad_bias = unbroadcast(g, bias.shape)
    return grad_x, grad_weight, grad_bias
