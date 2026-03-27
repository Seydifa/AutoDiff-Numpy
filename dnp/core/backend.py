"""
dnp/core/backend.py
===================
Hardware-agnostic array backend for the ``dnp`` library.

Overview
--------
This module provides a single ``backend`` object that transparently delegates
NumPy-style calls to either **NumPy** (CPU) or **CuPy** (NVIDIA GPU), depending
on what is available at import time.  The rest of the library only ever imports
from this module — it never decides *which* array library to use on its own.

Key public API
--------------
``backend``
    The singleton :class:`BackendWrapper` instance.  Use it exactly like
    ``numpy``::

        from dnp.core.backend import backend
        x = backend.zeros((3, 3))          # numpy.zeros or cupy.zeros
        y = backend.random.randn(100)      # numpy/cupy random
        sp = backend.scipy.linalg          # scipy or cupyx.scipy.linalg

``is_cuda_available``
    ``True`` when CuPy was successfully imported.

``set_dtype(dtype)`` / ``get_dtype()``
    Get or set the global default floating-point dtype for all new Tensors.

``as_numpy(data)`` / ``as_cupy(data)`` / ``to_device(data, device)``
    Explicit CPU↔GPU array transfers.

``safe_eps(data)``
    Return a dtype-safe stability epsilon (never underflows to 0 in float16).

``synchronize()`` / ``get_device_count()``
    GPU synchronisation and device enumeration helpers.

Fallback behaviour
------------------
When CuPy is the active backend and a requested attribute does not exist on
``cupy`` (e.g. a rarely-implemented NumPy op), :class:`BackendWrapper` emits a
:class:`UserWarning` and automatically falls back to the NumPy implementation,
converting cupy arrays to/from numpy around the call so the rest of the code
stays device-transparent.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import warnings

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import numpy as np

try:
    import scipy as _scipy
except ImportError:  # pragma: no cover
    _scipy = None

# ---------------------------------------------------------------------------
# CUDA / CuPy detection  (must come before BackendWrapper so `cp` is in scope)
# ---------------------------------------------------------------------------
is_cuda_available = False
try:
    import cupy as cp
    import cupyx.scipy as _cupyx_scipy

    is_cuda_available = True
except ImportError:
    is_cuda_available = False
    _cupyx_scipy = None


# ---------------------------------------------------------------------------
# BackendWrapper
# ---------------------------------------------------------------------------
class BackendWrapper:
    """Thin proxy around a numpy-compatible module (numpy or cupy).

    All attribute lookups are forwarded to the wrapped module, so call sites
    such as ``backend.zeros(...)``, ``backend.random.randn(...)`` or
    ``backend.linalg.norm(...)`` work identically whether the active backend
    is NumPy or CuPy.

    Fallback mechanism
    ~~~~~~~~~~~~~~~~~~
    When the active backend is CuPy and the requested attribute is *not*
    present on ``cupy``, the wrapper:

    1. Emits a :class:`UserWarning` identifying the missing op.
    2. Looks up the same name on ``numpy``.
    3. Returns a thin ``_cpu_fallback`` callable that:
       - converts any ``cupy.ndarray`` arguments to ``numpy.ndarray`` via
         ``cp.asnumpy``,
       - runs the NumPy implementation, and
       - converts any ``numpy.ndarray`` result back to ``cupy.ndarray`` via
         ``cp.asarray``.

    scipy sub-package
    ~~~~~~~~~~~~~~~~~
    ``backend.scipy`` resolves to ``cupyx.scipy`` (CuPy backend) or ``scipy``
    (NumPy backend), giving access to GPU/CPU-accelerated sparse, signal, and
    linear-algebra routines through a single attribute.

    Parameters
    ----------
    module : module
        The primary array library to wrap (``numpy`` or ``cupy``).

    Examples
    --------
    >>> from dnp.core.backend import backend
    >>> backend.zeros(3)
    array([0., 0., 0.])
    >>> backend.scipy.linalg          # scipy.linalg or cupyx.scipy.linalg
    <module '...'>
    """

    def __init__(self, module):
        # Use object.__setattr__ to avoid triggering our own __setattr__
        object.__setattr__(self, "_module", module)

    # ------------------------------------------------------------------
    # Device control
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        """Return the currently active device: ``'cuda'`` or ``'cpu'``."""
        module = object.__getattribute__(self, "_module")
        return "cuda" if module is not np else "cpu"

    def set_device(self, device: str) -> None:
        """Switch the active backend module between CPU (NumPy) and GPU (CuPy).

        Parameters
        ----------
        device : {'cpu', 'cuda'}
            Target device string.
            - ``'cuda'`` switches the backend to CuPy.  Raises
              :class:`RuntimeError` when CuPy is not installed.
            - ``'cpu'``  switches the backend to NumPy, even when CuPy is
              available.  Useful for debugging or when a GPU is present but
              you want reproducible CPU execution.

        Raises
        ------
        RuntimeError
            If ``device='cuda'`` is requested but CuPy is not installed.
        ValueError
            If an unrecognised device string is given.

        Examples
        --------
        >>> from dnp.core.backend import backend
        >>> backend.set_device('cpu')   # force CPU even if GPU is available
        >>> backend.device
        'cpu'
        >>> backend.set_device('cuda')  # switch back to GPU
        """
        if device == "cuda":
            if not is_cuda_available:
                raise RuntimeError(
                    "CuPy is not installed — cannot switch to CUDA backend. "
                    "See https://docs.cupy.dev/en/stable/install.html"
                )
            object.__setattr__(self, "_module", cp)
        elif device == "cpu":
            object.__setattr__(self, "_module", np)
        else:
            raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")

    @property
    def scipy(self):
        """Return the scipy-compatible sub-package for this backend.

        Returns
        -------
        module
            ``cupyx.scipy`` when the active backend is CuPy, ``scipy``
            otherwise.

        Raises
        ------
        RuntimeError
            If the required scipy package is not installed.
        """
        module = object.__getattribute__(self, "_module")
        if module is not np:
            # CuPy backend
            if _cupyx_scipy is None:  # pragma: no cover
                raise RuntimeError(
                    "cupyx.scipy is not available. "
                    "It is bundled with CuPy — reinstall CuPy to fix this."
                )
            return _cupyx_scipy
        # NumPy backend
        if _scipy is None:  # pragma: no cover
            raise RuntimeError("scipy is not installed. Run: pip install scipy")
        return _scipy

    def __getattr__(self, name):
        module = object.__getattribute__(self, "_module")

        # Fast path: attribute exists on the primary backend
        if hasattr(module, name):
            return getattr(module, name)

        # Attribute is missing — warn if we were using CuPy
        if module is not np:
            warnings.warn(
                f"'{name}' is not available in the CUDA backend; "
                "falling back to NumPy CPU implementation.",
                stacklevel=2,
            )

        # Raises AttributeError naturally if numpy also lacks the attribute
        np_attr = getattr(np, name)

        # numpy was already the primary module — return directly
        if module is np:
            return np_attr

        # Build a wrapper that converts cupy→numpy on input and
        # numpy→cupy on output so callers stay on-device transparently.
        def _cpu_fallback(*args, **kwargs):
            def _to_cpu(x):
                if isinstance(x, cp.ndarray):
                    return cp.asnumpy(x)
                return x

            def _to_gpu(r):
                """Recursively convert numpy arrays in a result back to cupy."""
                if isinstance(r, np.ndarray):
                    return cp.asarray(r)
                if isinstance(r, tuple):
                    return tuple(_to_gpu(item) for item in r)
                if isinstance(r, list):
                    return [_to_gpu(item) for item in r]
                return r

            cpu_args = tuple(_to_cpu(a) for a in args)
            cpu_kwargs = {k: _to_cpu(v) for k, v in kwargs.items()}
            result = np_attr(*cpu_args, **cpu_kwargs)
            return _to_gpu(result)

        return _cpu_fallback

    def __repr__(self):
        mod = object.__getattribute__(self, "_module")
        return f"BackendWrapper(device='{self.device}', module={mod.__name__})"


# ---------------------------------------------------------------------------
# Singleton backend instance
# ---------------------------------------------------------------------------
# Always start on CPU (NumPy).  Call backend.set_device('cuda') or
# dnp.set_device('cuda') to switch to CuPy.  Auto-switching based on
# CUDA availability would capture CuPy ufuncs inside Ops at import time,
# making CPU mode fail later when set_device('cpu') is called.
backend = BackendWrapper(np)

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


# ---------------------------------------------------------------------------
# Array transfer utilities
# ---------------------------------------------------------------------------
def is_cupy_array(data):
    """Return ``True`` if *data* is a CuPy ndarray.

    Parameters
    ----------
    data : object
        Any object to test.

    Returns
    -------
    bool
        ``True`` when CuPy is available and *data* is an instance of
        ``cupy.ndarray``; ``False`` otherwise.
    """
    if is_cuda_available:
        return isinstance(data, cp.ndarray)
    return False


def get_xp(data):
    """Return the array-library module that owns *data*.

    Parameters
    ----------
    data : array-like
        A numpy or cupy array (or anything with compatible dtype semantics).

    Returns
    -------
    module
        ``cupy`` if *data* is a ``cupy.ndarray`` and CuPy is available;
        ``numpy`` in all other cases.
    """
    if is_cuda_available and isinstance(data, cp.ndarray):
        return cp
    return np


def as_numpy(data):
    """Return *data* as a ``numpy.ndarray``, transferring from GPU if needed.

    Parameters
    ----------
    data : array-like
        A numpy array, cupy array, or any object accepted by
        ``numpy.asarray``.

    Returns
    -------
    numpy.ndarray
        CPU copy of *data*.
    """
    if is_cuda_available and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return np.asarray(data)


def as_cupy(data):
    """Return *data* as a ``cupy.ndarray``, transferring from CPU if needed.

    Parameters
    ----------
    data : array-like
        A numpy array, cupy array, or any object accepted by
        ``cupy.asarray``.

    Returns
    -------
    cupy.ndarray
        GPU copy of *data*.

    Raises
    ------
    RuntimeError
        If CuPy is not installed.
    """
    if not is_cuda_available:
        raise RuntimeError(
            "CuPy is not available. Install it for GPU support: "
            "https://docs.cupy.dev/en/stable/install.html"
        )
    if isinstance(data, cp.ndarray):
        return data
    return cp.asarray(data)


def to_device(data, device: str):
    """Transfer *data* to the requested device.

    Parameters
    ----------
    data : array-like
        Source array (numpy or cupy).
    device : {'cpu', 'cuda'}
        Target device string.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        Array residing on the requested device.

    Raises
    ------
    RuntimeError
        If ``device='cuda'`` but CuPy is not installed.
    """
    if device == "cuda":
        return as_cupy(data)
    return as_numpy(data)


def safe_eps(data) -> float:
    """Return a dtype-safe numerical stability epsilon for *data*.

    The hard-coded constant ``1e-8`` used as a division guard underflows to
    **zero** in float16 (the smallest positive subnormal is ~5.96e-8), making
    expressions like ``x + 1e-8`` no-ops and causing division-by-zero during
    float16 training.  This function returns ``numpy.finfo(dtype).tiny`` —
    the smallest *representable* positive normal for the dtype — which is
    always non-zero:

    +-----------+--------------+
    | dtype     | tiny value   |
    +===========+==============+
    | float16   | 6.10e-05     |
    +-----------+--------------+
    | float32   | 1.18e-38     |
    +-----------+--------------+
    | float64   | 2.23e-308    |
    +-----------+--------------+

    Parameters
    ----------
    data : array-like with a ``.dtype`` attribute
        Any numpy / cupy array or Tensor whose dtype is used to determine
        the appropriate epsilon.

    Returns
    -------
    float
        A Python float safe to add to arrays of *data*'s dtype without
        underflowing to zero.  Falls back to ``1e-8`` for integer dtypes.

    Examples
    --------
    >>> import numpy as np
    >>> from dnp.core.backend import safe_eps
    >>> safe_eps(np.zeros(1, dtype=np.float16))
    6.103515625e-05
    >>> safe_eps(np.zeros(1, dtype=np.float32))
    1.1754943508222875e-38
    """
    dtype = getattr(data, "dtype", np.float64)
    if np.issubdtype(dtype, np.floating):
        return float(np.finfo(dtype).tiny)
    return 1e-8


# ---------------------------------------------------------------------------
# Device / GPU utilities
# ---------------------------------------------------------------------------
def synchronize():
    """Block until all pending GPU operations on the default stream are done.

    Calls ``cupy.cuda.Stream.null.synchronize()`` when CuPy is available;
    otherwise this is a no-op.  Useful for accurate wall-clock benchmarking of
    GPU kernels, since CUDA operations are launched asynchronously by default.

    Examples
    --------
    >>> import time
    >>> from dnp.core.backend import backend, synchronize
    >>> t0 = time.perf_counter()
    >>> _ = backend.zeros((1024, 1024))
    >>> synchronize()          # ensure GPU work is done before stopping timer
    >>> elapsed = time.perf_counter() - t0
    """
    if is_cuda_available:
        cp.cuda.Stream.null.synchronize()


def get_device_count() -> int:
    """Return the number of available CUDA-capable devices.

    Returns
    -------
    int
        Number of CUDA devices visible to CuPy.  Returns ``0`` when CuPy is
        not installed or no CUDA devices are present.
    """
    if not is_cuda_available:
        return 0
    return cp.cuda.runtime.getDeviceCount()


def set_device(device: str) -> None:
    """Switch the active compute device for the global ``backend`` singleton.

    A convenience wrapper around :meth:`BackendWrapper.set_device` so callers
    can use a simple module-level import::

        from dnp.core.backend import set_device
        set_device('cpu')   # force CPU even when a GPU is present
        set_device('cuda')  # switch back to GPU

    Parameters
    ----------
    device : {'cpu', 'cuda'}
        Target device string.

    Raises
    ------
    RuntimeError
        If ``'cuda'`` is requested but CuPy is not installed.
    ValueError
        If an unrecognised device string is given.
    """
    backend.set_device(device)
