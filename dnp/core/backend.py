import numpy as np

is_cuda_available = False
try:
    import cupy as cp
    is_cuda_available = True
    backend = cp
except ImportError:
    is_cuda_available = False
    backend = np

def get_xp(data):
    """Return the correct module (numpy or cupy) based on the array type."""
    if is_cuda_available:
        return cp.get_array_module(data)
    return np

def as_numpy(data):
    """Ensure data is a numpy array."""
    if is_cuda_available and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return np.asarray(data)

def as_cupy(data):
    """Ensure data is a cupy array, if available."""
    if not is_cuda_available:
        raise RuntimeError("CuPy is not available on this system.")
    if isinstance(data, cp.ndarray):
        return data
    return cp.asarray(data)
