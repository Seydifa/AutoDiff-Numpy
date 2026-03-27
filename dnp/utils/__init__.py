"""dnp.utils — training utilities."""

from .trainer import (
    Callback,
    EarlyStopping,
    History,
    ModelCheckpoint,
    ProgressLogger,
    Trainer,
)


def device_op_percentage(device: str) -> float:
    """Return the percentage of forward-pass operations that ran on *device*.

    Tracks every call made through any :class:`~dnp.core.ops.Ops` instance
    since the last :func:`reset_device_stats` call (or since program start).

    Parameters
    ----------
    device : {'cpu', 'cuda'}
        The device to query.

    Returns
    -------
    float
        Percentage in ``[0.0, 100.0]``.  Returns ``0.0`` when no operations
        have been recorded yet.

    Examples
    --------
    >>> import dnp
    >>> from dnp.utils import device_op_percentage, reset_device_stats
    >>> reset_device_stats()
    >>> x = dnp.ops.add(dnp.ops.ones((3,)), dnp.ops.zeros((3,)))
    >>> device_op_percentage('cpu')
    100.0
    """
    from dnp.core.ops import device_stats

    return device_stats.percentage(device)


def reset_device_stats() -> None:
    """Reset the per-device operation counters to zero.

    Call this before a code section you want to profile so that counts from
    earlier runs do not skew the result.
    """
    from dnp.core.ops import device_stats

    device_stats.reset()


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """Clip the gradient norm of an iterable of parameters in-place.

    The gradients are modified in place.  Based on PyTorch's
    ``torch.nn.utils.clip_grad_norm_``.

    Parameters
    ----------
    parameters : iterable of Tensor
        Model parameters whose ``.grad`` buffers will be clipped.
    max_norm : float
        Maximum allowed total gradient norm.
    norm_type : float
        Type of the norm (e.g. 2.0 for L2, 1.0 for L1, float('inf') for max).

    Returns
    -------
    float
        The total gradient norm before clipping.
    """
    import math
    import numpy as np
    from dnp.core.backend import as_numpy

    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    if norm_type == float("inf"):
        total_norm = max(float(np.max(np.abs(as_numpy(p.grad)))) for p in params)
    else:
        total_norm = float(
            sum(float(np.sum(np.abs(as_numpy(p.grad)) ** norm_type)) for p in params)
            ** (1.0 / norm_type)
        )

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad *= clip_coef

    return total_norm


def clip_grad_value_(parameters, clip_value: float) -> None:
    """Clip all gradient values element-wise to ``[-clip_value, clip_value]``.

    Parameters
    ----------
    parameters : iterable of Tensor
        Model parameters whose ``.grad`` buffers will be clipped.
    clip_value : float
        Maximum absolute value for any gradient element.
    """
    from dnp.core.backend import backend

    for p in parameters:
        if p.grad is not None:
            p.grad = backend.clip(p.grad, -clip_value, clip_value)


class DataLoader:
    """Simple mini-batch iterator over numpy arrays.

    Parameters
    ----------
    dataset : tuple of array-like
        One or more arrays of the same length (e.g. ``(X, y)``).  All arrays
        are iterated together; each batch is a tuple of slices.
    batch_size : int
        Number of samples per mini-batch.
    shuffle : bool
        Shuffle the data before each epoch when ``True``.
    drop_last : bool
        When ``True``, drop the final batch if it is smaller than
        ``batch_size``.

    Examples
    --------
    >>> loader = DataLoader((X_train, y_train), batch_size=32, shuffle=True)
    >>> for epoch in range(10):
    ...     for X_batch, y_batch in loader:
    ...         ...
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        import numpy as np

        if not isinstance(dataset, (list, tuple)):
            dataset = (dataset,)
        self.dataset = tuple(np.asarray(d) for d in dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._n = len(self.dataset[0])

    def __len__(self) -> int:
        """Number of batches per epoch."""
        import math

        if self.drop_last:
            return self._n // self.batch_size
        return math.ceil(self._n / self.batch_size)

    def __iter__(self):
        import numpy as np

        idx = np.arange(self._n)
        if self.shuffle:
            np.random.shuffle(idx)
        for start in range(0, self._n, self.batch_size):
            end = start + self.batch_size
            if self.drop_last and end > self._n:
                break
            batch_idx = idx[start:end]
            yield tuple(arr[batch_idx] for arr in self.dataset)


__all__ = [
    "Callback",
    "EarlyStopping",
    "History",
    "ModelCheckpoint",
    "ProgressLogger",
    "Trainer",
    "DataLoader",
    "device_op_percentage",
    "reset_device_stats",
    "clip_grad_norm_",
    "clip_grad_value_",
]
