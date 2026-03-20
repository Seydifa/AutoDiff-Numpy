"""
dnp/core/vjp_rules.py
=====================
Vector-Jacobian Product (VJP) rules for all supported operations,
plus activation functions and convolution helpers.

This is the single source of truth — previously split across dnp/ops/vjp_rules.py.
"""

# Third-party libraries
import numpy as np
import numba

from .backend import get_xp, as_numpy

EPSILON = 1e-8

# ---------------------------------------------------------------------------
# Numba-optimized Kernels (CPU bound, so input needs to be transferred)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _max_pool2d_backward_kernel(g, x_padded, kH, kW, sH, sW, H_out, W_out):
    batch, channels, _, _ = x_padded.shape
    grad_x = np.zeros_like(x_padded)
    for b in range(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * sH
                    h_end = h_start + kH
                    w_start = j * sW
                    w_end = w_start + kW
                    
                    max_val = -1e30 
                    for wh in range(h_start, h_end):
                        for ww in range(w_start, w_end):
                            if x_padded[b, c, wh, ww] > max_val:
                                max_val = x_padded[b, c, wh, ww]
                    
                    for wh in range(h_start, h_end):
                        for ww in range(w_start, w_end):
                            if x_padded[b, c, wh, ww] == max_val:
                                grad_x[b, c, wh, ww] += g[b, c, i, j]
    return grad_x


@numba.njit(cache=True)
def _avg_pool2d_backward_kernel(g, x_padded, kH, kW, sH, sW, H_out, W_out):
    batch, channels, _, _ = x_padded.shape
    grad_x = np.zeros_like(x_padded)
    pool_size = kH * kW
    for b in range(batch):
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * sH
                    h_end = h_start + kH
                    w_start = j * sW
                    w_end = w_start + kW
                    
                    val = g[b, c, i, j] / pool_size
                    for wh in range(h_start, h_end):
                        for ww in range(w_start, w_end):
                            grad_x[b, c, wh, ww] += val
    return grad_x


@numba.njit(cache=True)
def _conv2d_forward_kernel(x_padded, W, stride_h, stride_w, H_out, W_out):
    batch_size, in_channels, _, _ = x_padded.shape
    out_channels, _, kH, kW = W.shape
    y = np.zeros((batch_size, out_channels, H_out, W_out), dtype=x_padded.dtype)
    
    for b in range(batch_size):
        for oc in range(out_channels):
            for ic in range(in_channels):
                for i in range(H_out):
                    h_start = i * stride_h
                    for j in range(W_out):
                        w_start = j * stride_w
                        
                        sum_val = 0.0
                        for kh in range(kH):
                            for kw in range(kW):
                                sum_val += x_padded[b, ic, h_start + kh, w_start + kw] * W[oc, ic, kh, kw]
                        y[b, oc, i, j] += sum_val
    return y


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
    x_clipped = xp.clip(x, -500, 500)
    return xp.log(1.0 + xp.exp(x_clipped))

def swish(x):
    return x * sigmoid(x)

def gelu(x):
    xp = get_xp(x)
    const1 = xp.sqrt(2.0 / xp.pi)
    const2 = 0.044715
    return 0.5 * x * (1.0 + xp.tanh(const1 * (x + const2 * xp.power(x, 3))))

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
    is_cupy = (xp.__name__ == 'cupy')
    if is_cupy:
        x_np = as_numpy(x)
    else:
        x_np = x

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    batch, channels, H, W = x_np.shape
    kH, kW = kernel_size
    sH, sW = stride

    if padding > 0:
        x_np = np.pad(x_np, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        H, W = x_np.shape[2:]

    H_out = (H - kH) // sH + 1
    W_out = (W - kW) // sW + 1

    strides = x_np.strides
    shape = (batch, channels, H_out, W_out, kH, kW)
    strides = (strides[0], strides[1], sH * strides[2], sW * strides[3], strides[2], strides[3])

    x_windowed = np.lib.stride_tricks.as_strided(x_np, shape=shape, strides=strides)
    out = np.max(x_windowed, axis=(4, 5))

    if is_cupy:
        return xp.asarray(out)
    return out

def avg_pool2d(x, kernel_size, stride=1, padding=0):
    xp = get_xp(x)
    is_cupy = (xp.__name__ == 'cupy')
    if is_cupy:
        x_np = as_numpy(x)
    else:
        x_np = x

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    batch, channels, H, W = x_np.shape
    kH, kW = kernel_size
    sH, sW = stride

    if padding > 0:
        x_np = np.pad(x_np, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        H, W = x_np.shape[2:]

    H_out = (H - kH) // sH + 1
    W_out = (W - kW) // sW + 1

    strides = x_np.strides
    shape = (batch, channels, H_out, W_out, kH, kW)
    strides = (strides[0], strides[1], sH * strides[2], sW * strides[3], strides[2], strides[3])

    x_windowed = np.lib.stride_tricks.as_strided(x_np, shape=shape, strides=strides)
    out = np.mean(x_windowed, axis=(4, 5))
    if is_cupy:
        return xp.asarray(out)
    return out

def dropout(x, p=0.5, training=True):
    if not training or p == 0:
        return x
    xp = get_xp(x)
    mask = xp.random.binomial(1, 1 - p, size=x.shape)
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
    if xp.__name__ == 'cupy':
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
        unbroadcast(get_xp(g).matmul(g, get_xp(y).swapaxes(y, -1, -2) if getattr(y, 'ndim', 0) >= 2 else y), x.shape),
        unbroadcast(get_xp(g).matmul(get_xp(x).swapaxes(x, -1, -2) if getattr(x, 'ndim', 0) >= 2 else x, g), y.shape),
    ),
    np.dot: lambda g, x, y: (get_xp(g).dot(g, y.T), get_xp(g).dot(x.T, g)),
    # Reductions
    np.sum: lambda g, x, axis=None, keepdims=False: (
        (g if keepdims else get_xp(g).expand_dims(g, axis) if axis is not None else g)
        * get_xp(x).ones_like(x),
    ),
    np.mean: lambda g, x, axis=None, keepdims=False: (
        (
            (g if keepdims else get_xp(g).expand_dims(g, axis) if axis is not None else g)
            * get_xp(x).ones_like(x)
        )
        / (get_xp(x).prod(get_xp(x).array(x.shape)[axis]) if axis is not None else x.size),
    ),
    np.prod: lambda g, x, axis=None, keepdims=False: (
        (g if keepdims else get_xp(g).expand_dims(g, axis) if axis is not None else g)
        * (get_xp(x).prod(x, axis=axis, keepdims=True) / (x + EPSILON)),
    ),
    np.max: lambda g, x, axis=None, keepdims=False: (
        (g if keepdims else get_xp(g).expand_dims(g, axis) if axis is not None else g)
        * (x == get_xp(x).max(x, axis=axis, keepdims=True)),
    ),
    np.min: lambda g, x, axis=None, keepdims=False: (
        (g if keepdims else get_xp(g).expand_dims(g, axis) if axis is not None else g)
        * (x == get_xp(x).min(x, axis=axis, keepdims=True)),
    ),
    # Shape operations
    np.reshape: lambda g, x, newshape: (get_xp(g).reshape(g, x.shape),),
    np.transpose: lambda g, x, axes=None: (
        get_xp(g).transpose(g, get_xp(g).argsort(axes)) if axes is not None else g.T,
    ),
    np.expand_dims: lambda g, x, axis: (get_xp(g).squeeze(g, axis),),
    np.squeeze: lambda g, x, axis=None: (
        get_xp(g).expand_dims(g, axis) if axis is not None else get_xp(g).reshape(g, x.shape),
    ),
    # NN ops
    conv2d: lambda g, x, w, mode="valid": (
        conv2d(g, rot180(w), mode="full" if mode == "valid" else "same" if mode == "same" else "valid"),
        conv2d(
            x if mode == "valid" else
            get_xp(x).pad(x, [(w.shape[0] // 2,) * 2, (w.shape[1] // 2,) * 2]) if mode == "same" else
            get_xp(x).pad(x, [(w.shape[0] - 1,) * 2, (w.shape[1] - 1,) * 2]),
            g, mode="valid"
        ),
    ),
    sigmoid: lambda g, x: (g * sigmoid(x) * (1.0 - sigmoid(x)),),
    relu: lambda g, x: (g * (x > 0).astype(x.dtype),),
    leaky_relu: lambda g, x: (g * get_xp(x).where(x > 0, 1.0, 0.01),),
    elu: lambda g, x: (g * get_xp(x).where(x > 0, 1.0, 1.0 * get_xp(x).exp(x)),),
    softplus: lambda g, x: (g * sigmoid(x),),
    swish: lambda g, x: (g * (swish(x) + sigmoid(x) * (1.0 - swish(x))),),
    gelu: lambda g, x: (
        g * (
            0.5 * (1.0 + get_xp(x).tanh(get_xp(x).sqrt(2 / get_xp(x).pi) * (x + 0.044715 * x**3)))
            + 0.5 * x * (1.0 - get_xp(x).tanh(get_xp(x).sqrt(2 / get_xp(x).pi) * (x + 0.044715 * x**3)) ** 2)
            * get_xp(x).sqrt(2 / get_xp(x).pi) * (1.0 + 3 * 0.044715 * x**2)
        ),
    ),
    softmax: lambda g, x: (g * softmax(x) - softmax(x) * get_xp(x).sum(g * softmax(x), axis=-1, keepdims=True),),
    max_pool2d: lambda g, x, kernel_size, stride=1, padding=0: (_max_pool2d_backward(g, x, kernel_size, stride, padding),),
    avg_pool2d: lambda g, x, kernel_size, stride=1, padding=0: (_avg_pool2d_backward(g, x, kernel_size, stride, padding),),
    dropout: lambda g, x, p=0.5, training=True: ((g / (1 - p),) if training else (g,)),
    batch_norm: lambda g, x, weight, bias, mean, var, eps=1e-5: (_batch_norm_backward(g, x, weight, bias, mean, var, eps),),
}

# ---------------------------------------------------------------------------
# Backward pass helpers
# ---------------------------------------------------------------------------

def _max_pool2d_backward(g, x, kernel_size, stride=1, padding=0):
    xp = get_xp(x)
    is_cpu = (xp.__name__ == 'numpy')
    g_np, x_np = as_numpy(g), as_numpy(x)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    batch, channels, H, W = x_np.shape
    kH, kW = kernel_size
    sH, sW = stride

    if padding > 0:
        x_padded = np.pad(x_np, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    else:
        x_padded = x_np

    H_padded, W_padded = x_padded.shape[2:]
    H_out = (H_padded - kH) // sH + 1
    W_out = (W_padded - kW) // sW + 1

    grad_x_padded = _max_pool2d_backward_kernel(g_np, x_padded, kH, kW, sH, sW, H_out, W_out)

    if padding > 0:
        grad_x = grad_x_padded[:, :, padding:-padding, padding:-padding]
    else:
        grad_x = grad_x_padded

    return grad_x if is_cpu else xp.asarray(grad_x)

def _avg_pool2d_backward(g, x, kernel_size, stride=1, padding=0):
    xp = get_xp(x)
    is_cpu = (xp.__name__ == 'numpy')
    g_np, x_np = as_numpy(g), as_numpy(x)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    batch, channels, H, W = x_np.shape
    kH, kW = kernel_size
    sH, sW = stride

    if padding > 0:
        x_padded = np.pad(x_np, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    else:
        x_padded = x_np

    H_padded, W_padded = x_padded.shape[2:]
    H_out = (H_padded - kH) // sH + 1
    W_out = (W_padded - kW) // sW + 1

    grad_x_padded = _avg_pool2d_backward_kernel(g_np, x_padded, kH, kW, sH, sW, H_out, W_out)

    if padding > 0:
        grad_x = grad_x_padded[:, :, padding:-padding, padding:-padding]
    else:
        grad_x = grad_x_padded

    return grad_x if is_cpu else xp.asarray(grad_x)

def _batch_norm_backward(g, x, weight, bias, mean, var, eps=1e-5):
    xp = get_xp(x)
    x_norm = (x - mean) / xp.sqrt(var + eps)
    grad_x_norm = g * weight
    N = x.shape[0]
    grad_x = (
        grad_x_norm / xp.sqrt(var + eps)
        - (grad_x_norm * x_norm) * (1 / (var + eps)) * (x - mean) / N
        - xp.mean(grad_x_norm, axis=0, keepdims=True) / xp.sqrt(var + eps)
    )
    return grad_x
