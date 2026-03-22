import numpy as np

is_cuda_available = False
try:
    import cupy as cp

    is_cuda_available = True
    backend = cp
except ImportError:
    is_cuda_available = False
    backend = np

# ---------------------------------------------------------------------------
# Global dtype registry
# ---------------------------------------------------------------------------
_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}

_DEFAULT_DTYPE = np.float64


def set_dtype(dtype) -> None:
    """Set the global default floating-point dtype used by all new Tensors.

    Parameters
    ----------
    dtype : str or numpy dtype
        ``'float16'``, ``'float32'``, or ``'float64'`` (default).
        A numpy/cupy dtype object is also accepted directly.

    Examples
    --------
    >>> import dnp
    >>> dnp.core.backend.set_dtype('float32')   # all new Tensors use float32
    >>> dnp.core.backend.set_dtype('float64')   # revert to default
    """
    global _DEFAULT_DTYPE
    if isinstance(dtype, str):
        if dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Choose from {list(_DTYPE_MAP)}."
            )
        _DEFAULT_DTYPE = _DTYPE_MAP[dtype]
    else:
        # Accept numpy/cupy dtype objects directly
        _DEFAULT_DTYPE = np.dtype(dtype).type


def get_dtype():
    """Return the current global default floating-point dtype.

    Returns
    -------
    numpy dtype type (e.g. ``numpy.float64``)
    """
    return _DEFAULT_DTYPE


def is_cupy_array(data):
    """Return True if *data* is a CuPy ndarray."""
    if is_cuda_available:
        return isinstance(data, cp.ndarray)
    return False


def get_xp(data):
    """Return the correct module (numpy or cupy) based on the array type."""
    if is_cuda_available and type(data).__module__ == "cupy":
        return cp
    return np


def as_numpy(data):
    """Ensure data is a numpy array (transfers from GPU if necessary)."""
    if is_cuda_available and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return np.asarray(data)


def as_cupy(data):
    """Ensure data is a cupy array (transfers from CPU if necessary)."""
    if not is_cuda_available:
        raise RuntimeError(
            "CuPy is not available. Install it for GPU support: "
            "https://docs.cupy.dev/en/stable/install.html"
        )
    if isinstance(data, cp.ndarray):
        return data
    return cp.asarray(data)


def to_device(data, device: str):
    """Transfer *data* to *device* ('cpu' or 'cuda')."""
    if device == "cuda":
        return as_cupy(data)
    return as_numpy(data)


def safe_eps(data) -> float:
    """Return a dtype-safe numerical stability epsilon for *data*.

    The hard-coded constant ``1e-8`` used as a division guard underflows to
    **zero** in float16 (the smallest subnormal is ~5.96e-8), making guards
    like ``x + 1e-8`` no-ops and causing division-by-zero during float16
    training.  This function instead returns ``finfo(dtype).tiny`` — the
    smallest *representable* positive normal for the dtype — which is always
    non-zero:

    * float16 → 6.10e-05
    * float32 → 1.18e-38
    * float64 → 2.23e-308

    Parameters
    ----------
    data : array-like with a ``.dtype`` attribute
        Any numpy / cupy array or Tensor whose dtype should be used.

    Returns
    -------
    float
        A Python float safe to add to arrays of *data*'s dtype without
        underflowing to zero.
    """
    dtype = getattr(data, "dtype", np.float64)
    if np.issubdtype(dtype, np.floating):
        return float(np.finfo(dtype).tiny)
    return 1e-8


def synchronize():
    """Block until all pending GPU operations are complete.

    This is a no-op when CUDA is not available.  Useful for accurate
    timing of GPU kernels.
    """
    if is_cuda_available:
        cp.cuda.Stream.null.synchronize()


def get_device_count() -> int:
    """Return the number of available CUDA devices (0 if none)."""
    if not is_cuda_available:
        return 0
    return cp.cuda.runtime.getDeviceCount()
