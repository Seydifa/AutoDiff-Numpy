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


__all__ = [
    "Callback",
    "EarlyStopping",
    "History",
    "ModelCheckpoint",
    "ProgressLogger",
    "Trainer",
    "device_op_percentage",
    "reset_device_stats",
]
