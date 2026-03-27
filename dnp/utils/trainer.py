"""
dnp.utils.trainer
=================

High-level training loop with a callback system and model persistence.

Usage
-----
>>> from dnp.utils import Trainer, EarlyStopping, ModelCheckpoint
>>> trainer = Trainer(model, optimizer, loss_fn,
...                   callbacks=[EarlyStopping(patience=5),
...                               ModelCheckpoint("best.npz")])
>>> trainer.fit(X_train, y_train, epochs=50, batch_size=32)
>>> y_pred = trainer.predict(X_test)
"""

from __future__ import annotations

import pickle
import time
import warnings
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from dnp.core.backend import as_numpy


# ---------------------------------------------------------------------------
# Callback base class
# ---------------------------------------------------------------------------


class Callback:
    """Base class for all callbacks.

    All hook methods receive a ``logs`` dict that may contain keys such as
    ``epoch``, ``loss``, ``val_loss``, ``batch``, etc.

    Subclasses override only the hooks they need.
    """

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Called once before training starts."""

    def on_train_end(self, logs: dict | None = None) -> None:
        """Called once after training ends."""

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> bool | None:
        """Called at the end of each epoch.

        Return ``True`` to request early stopping.
        """

    def on_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        """Called at the start of each mini-batch."""

    def on_batch_end(self, batch: int, logs: dict | None = None) -> None:
        """Called at the end of each mini-batch."""


# ---------------------------------------------------------------------------
# Built-in callbacks
# ---------------------------------------------------------------------------


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    monitor : str
        Metric name to watch.  Defaults to ``"val_loss"`` (falls back to
        ``"loss"`` if no validation data is provided).
    patience : int
        Epochs with no improvement after which training is stopped.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : ``"min"`` | ``"max"`` | ``"auto"``
        Whether lower or higher is better.  ``"auto"`` infers from the
        metric name (``"loss"`` → ``"min"``, else ``"max"``).
    restore_best_weights : bool
        Whether to restore model weights from the epoch with the best value.
    verbose : bool
        Print a message when stopping early.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "auto",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        if mode == "auto":
            self.mode = "min" if "loss" in monitor else "max"
        else:
            self.mode = mode

        self._best: float | None = None
        self._wait: int = 0
        self._best_weights: list | None = None

    # ------------------------------------------------------------------ #

    def _is_better(self, current: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return current < self._best - self.min_delta
        return current > self._best + self.min_delta

    # ------------------------------------------------------------------ #

    def on_train_begin(self, logs: dict | None = None) -> None:
        self._best = None
        self._wait = 0
        self._best_weights = None

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> bool | None:
        logs = logs or {}
        value = logs.get(self.monitor, logs.get("loss"))
        if value is None:
            return None

        if self._is_better(value):
            self._best = value
            self._wait = 0
            if self.restore_best_weights and "model" in logs:
                self._best_weights = _copy_weights(logs["model"])
        else:
            self._wait += 1
            if self._wait >= self.patience:
                if self.verbose:
                    print(
                        f"EarlyStopping: no improvement in '{self.monitor}' "
                        f"for {self.patience} epochs. Stopping at epoch {epoch + 1}."
                    )
                if (
                    self.restore_best_weights
                    and self._best_weights is not None
                    and "model" in logs
                ):
                    _restore_weights(logs["model"], self._best_weights)
                    if self.verbose:
                        print("EarlyStopping: restored best weights.")
                return True  # signal stop
        return None


class ModelCheckpoint(Callback):
    """Save the model after every epoch (or only when it improves).

    Parameters
    ----------
    filepath : str
        Path where the model will be saved.  ``.pkl`` and ``.npz`` are
        supported.  A ``.pkl`` file is always written (via pickle); a
        ``.npz`` file stores only the raw numpy arrays.
    monitor : str
        Metric to monitor when ``save_best_only=True``.
    save_best_only : bool
        If ``True`` only save when the monitored metric improves.
    mode : ``"min"`` | ``"max"`` | ``"auto"``
    verbose : bool
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "auto",
        verbose: bool = True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose

        if mode == "auto":
            self.mode = "min" if "loss" in monitor else "max"
        else:
            self.mode = mode

        self._best: float | None = None

    # ------------------------------------------------------------------ #

    def _is_better(self, current: float) -> bool:
        if self._best is None:
            return True
        return current < self._best if self.mode == "min" else current > self._best

    # ------------------------------------------------------------------ #

    def on_train_begin(self, logs: dict | None = None) -> None:
        self._best = None

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        model = logs.get("model")
        if model is None:
            return

        value = logs.get(self.monitor, logs.get("loss"))
        should_save = not self.save_best_only or (
            value is not None and self._is_better(value)
        )

        if should_save:
            if value is not None and self.save_best_only:
                self._best = value
            _save_model(model, self.filepath)
            if self.verbose:
                metric_str = (
                    f" — {self.monitor}: {value:.6f}" if value is not None else ""
                )
                print(f"ModelCheckpoint: saved model to '{self.filepath}'{metric_str}")


class ProgressLogger(Callback):
    """Print epoch metrics to stdout.

    Parameters
    ----------
    metrics : sequence of str
        Metrics to display.  If ``None``, all available keys are shown.
    """

    def __init__(self, metrics: Sequence[str] | None = None):
        self.metrics = metrics
        self._epoch_start: float = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        elapsed = time.time() - self._epoch_start
        keys = self.metrics if self.metrics else [k for k in logs if k != "model"]
        parts = [f"{k}: {logs[k]:.6f}" for k in keys if k in logs]
        print(f"Epoch {epoch + 1:4d} [{elapsed:.1f}s]  " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """High-level training harness.

    Parameters
    ----------
    model : dnp `Module`
        The model to train.
    optimizer : dnp `Optimizer`
        Pre-constructed optimizer (already wrapping ``model.parameters``).
    loss_fn : callable
        Function ``(y_pred, y_true) -> scalar Tensor``.
    scheduler : optional LRScheduler
        Learning-rate scheduler; its ``step()`` is called after each epoch.
    callbacks : list of Callback
        Callbacks executed during training.
    verbose : bool
        If ``True`` a default :class:`ProgressLogger` is prepended to the
        callback list automatically (unless one is already present).
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn: Callable,
        scheduler=None,
        callbacks: List[Callback] | None = None,
        verbose: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self._callbacks: List[Callback] = list(callbacks or [])
        if verbose and not any(isinstance(c, ProgressLogger) for c in self._callbacks):
            self._callbacks.insert(0, ProgressLogger())

    # ------------------------------------------------------------------ helpers

    def _call_callbacks(self, hook: str, *args, **kwargs) -> bool:
        """Call *hook* on all callbacks; return True if any requests stop."""
        stop = False
        for cb in self._callbacks:
            result = getattr(cb, hook)(*args, **kwargs)
            if result is True:
                stop = True
        return stop

    # ------------------------------------------------------------------ public

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_data: tuple | None = None,
        shuffle: bool = True,
    ) -> "History":
        """Train the model.

        Parameters
        ----------
        x, y : numpy arrays
            Training inputs and targets.
        epochs : int
            Number of full passes over the data.
        batch_size : int
            Mini-batch size.
        validation_data : tuple ``(x_val, y_val)``, optional
            If provided, validation loss is computed at the end of each epoch.
        shuffle : bool
            Shuffle data before each epoch.

        Returns
        -------
        History
            Object with a ``history`` dict mapping metric names to lists.
        """
        from ..core.session import session

        history = History()
        n_samples = x.shape[0]

        self._call_callbacks("on_train_begin", {})

        for epoch in range(epochs):
            self._call_callbacks("on_epoch_begin", epoch, {})

            if shuffle:
                idx = np.random.permutation(n_samples)
                x, y = x[idx], y[idx]

            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, n_samples, batch_size):
                batch_x = x[batch_start : batch_start + batch_size]
                batch_y = y[batch_start : batch_start + batch_size]
                batch = batch_start // batch_size

                self._call_callbacks("on_batch_begin", batch, {})

                # --- forward ---
                self.optimizer.zero_grad()
                with session.graph():
                    from ..core.tensor import Tensor

                    bx = Tensor(batch_x, name="x")
                    by = Tensor(batch_y, name="y")

                    y_pred = self.model(bx)
                    loss = self.loss_fn(y_pred, by)

                # --- backward ---
                loss.backward()
                self.optimizer.step()
                session.reset()

                batch_loss = float(loss)
                epoch_loss += batch_loss
                n_batches += 1

                self._call_callbacks("on_batch_end", batch, {"loss": batch_loss})

            avg_loss = epoch_loss / max(n_batches, 1)
            logs: dict = {"model": self.model, "loss": avg_loss}

            # --- validation ---
            if validation_data is not None:
                x_val, y_val = validation_data
                val_loss = self.evaluate(x_val, y_val)
                logs["val_loss"] = val_loss
                history.history.setdefault("val_loss", []).append(val_loss)

            history.history.setdefault("loss", []).append(avg_loss)

            if self.scheduler is not None:
                # ReduceLROnPlateau needs the metric value
                monitor_val = logs.get("val_loss", avg_loss)
                try:
                    # Try ReduceLROnPlateau-style (positional metric)
                    self.scheduler.step(monitor_val)
                except TypeError:
                    try:
                        self.scheduler.step()
                    except TypeError:
                        pass

            stop = self._call_callbacks("on_epoch_end", epoch, logs)
            if stop:
                break

        self._call_callbacks("on_train_end", {"history": history})
        return history

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 256) -> float:
        """Compute mean loss over the dataset without updating parameters.

        Returns
        -------
        float
            Mean loss over all batches.
        """
        from ..core.session import session
        from ..core.tensor import Tensor

        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, x.shape[0], batch_size):
            bx = x[batch_start : batch_start + batch_size]
            by = y[batch_start : batch_start + batch_size]

            with session.no_grad():
                bx_t = Tensor(bx, name="x")
                by_t = Tensor(by, name="y")
                y_pred = self.model(bx_t)
                loss = self.loss_fn(y_pred, by_t)

            total_loss += float(loss)
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def predict(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Run inference and return predictions as a numpy array.

        Parameters
        ----------
        x : numpy array
            Input data.
        batch_size : int
            Mini-batch size for inference.

        Returns
        -------
        numpy.ndarray
        """
        from ..core.session import session
        from ..core.tensor import Tensor

        outputs = []
        for batch_start in range(0, x.shape[0], batch_size):
            bx = x[batch_start : batch_start + batch_size]
            with session.no_grad():
                y_pred = self.model(Tensor(bx, name="x"))
            outputs.append(as_numpy(y_pred.data))

        return np.concatenate(outputs, axis=0)

    def save(self, path: str) -> None:
        """Persist the model to *path*.

        Parameters
        ----------
        path : str
            Destination file.  Use a ``.pkl`` or ``.npz`` extension.
        """
        _save_model(self.model, path)

    def load(self, path: str) -> None:
        """Load model weights from *path*.

        Parameters
        ----------
        path : str
            Source file previously created by :meth:`save`.
        """
        _load_model(self.model, path)


# ---------------------------------------------------------------------------
# History container
# ---------------------------------------------------------------------------


class History:
    """Stores per-epoch training metrics.

    Attributes
    ----------
    history : dict[str, list[float]]
        Mapping from metric name to list of recorded values (one per epoch).
    """

    def __init__(self):
        self.history: dict[str, list[float]] = {}

    def __repr__(self) -> str:
        keys = list(self.history)
        return f"History(metrics={keys}, epochs={len(next(iter(self.history.values()), []))})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _copy_weights(model) -> list:
    """Return a deep copy of all parameter arrays."""
    return [as_numpy(p.data).copy() for p in model.parameters()]


def _restore_weights(model, weights: list) -> None:
    """Overwrite parameter arrays with *weights* (in the same order)."""
    for p, w in zip(model.parameters(), weights):
        p[...] = w


def _save_model(model, path: str) -> None:
    """Save model to *path* as ``.npz`` (recommended) or ``.pkl`` (legacy).

    .. warning::
        The ``.pkl`` format uses Python's ``pickle`` module.  Only load
        ``.pkl`` checkpoints from **trusted sources** — a malicious file can
        execute arbitrary code during deserialization (CWE-502).
        Prefer the ``.npz`` format for all new code.
    """
    if path.endswith(".npz"):
        arrays = {
            f"param_{i}": as_numpy(p.data) for i, p in enumerate(model.parameters())
        }
        np.savez(path, **arrays)
    else:
        warnings.warn(
            f"Saving model as pickle ('{path}'). "
            "Pickle files can execute arbitrary code when loaded from untrusted sources. "
            "Use a '.npz' extension for safe weight-only serialization.",
            UserWarning,
            stacklevel=2,
        )
        with open(path, "wb") as fh:
            pickle.dump(
                {"weights": [as_numpy(p.data) for p in model.parameters()]},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def _load_model(model, path: str) -> None:
    """Load model weights from *path*.

    .. warning::
        Loading a ``.pkl`` checkpoint executes the pickled bytecode.  Only
        load files from **trusted sources** (CWE-502).  Prefer ``.npz``.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        for i, p in enumerate(model.parameters()):
            p[...] = data[f"param_{i}"]
    else:
        warnings.warn(
            f"Loading model from pickle ('{path}'). "
            "Only load pickle files from trusted sources — "
            "malicious files can execute arbitrary code (CWE-502). "
            "Prefer saving/loading with '.npz' instead.",
            UserWarning,
            stacklevel=2,
        )
        with open(path, "rb") as fh:
            checkpoint = pickle.load(fh)  # noqa: S301 — guarded by warning above
        for p, w in zip(model.parameters(), checkpoint["weights"]):
            p[...] = w
