"""
Loss functions.
"""

from .base import Module
from dnp.core.tensor import Tensor
from dnp.core.backend import safe_eps, to_device
from dnp.core import ops


def _validate_reduction(reduction: str) -> str:
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(
            f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
        )
    return reduction


def _ensure_tensor(value):
    return value if isinstance(value, Tensor) else Tensor(value)


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return ops.mean(loss)
    if reduction == "sum":
        return ops.sum(loss)
    return loss


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification."""

    def forward(self, logits: Tensor, targets) -> Tensor:
        shape = logits.shape
        if len(shape) > 2:
            B, T, V = shape[0], shape[1], shape[2]
            logits_flat = ops.reshape(logits, newshape=(B * T, V))
        else:
            logits_flat = logits

        if isinstance(targets, Tensor):
            targets_flat = ops.reshape(
                targets.detach().to(logits.device), newshape=(-1,)
            )
        else:
            targets_flat = to_device(targets, logits.device).reshape(-1)

        return ops.sparse_cce_with_logits_loss(logits_flat, targets_flat)


class MSELoss(Module):
    """Mean Squared Error loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = _validate_reduction(reduction)

    def forward(self, y_pred: Tensor, y_true) -> Tensor:
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        if self.reduction == "mean":
            return ops.mse_loss(y_pred, y_true)
        diff = y_pred - y_true
        sq = ops.square(diff)
        if self.reduction == "sum":
            return ops.sum(sq)
        return sq


class BCELoss(Module):
    """Binary Cross-Entropy loss (expects probabilities, not logits)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = _validate_reduction(reduction)

    def forward(self, y_pred: Tensor, y_true) -> Tensor:
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        if self.reduction == "mean":
            return ops.bce_loss(y_pred, y_true)
        eps = safe_eps(y_pred)
        pos = y_true * ops.log(y_pred + eps)
        neg = (1.0 - y_true) * ops.log(1.0 - y_pred + eps)
        loss = -(pos + neg)
        if self.reduction == "sum":
            return ops.sum(loss)
        return loss


class BCEWithLogitsLoss(Module):
    """Binary Cross-Entropy with logits."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = _validate_reduction(reduction)

    def forward(self, logits: Tensor, y_true) -> Tensor:
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        if self.reduction == "mean":
            return ops.bce_with_logits_loss(logits, y_true)
        relu_x = ops.relu(logits)
        abs_x = ops.absolute(logits)
        loss = relu_x - logits * y_true + ops.log1p(ops.exp(-abs_x))
        if self.reduction == "sum":
            return ops.sum(loss)
        return loss


class MAELoss(Module):
    """Mean Absolute Error loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = _validate_reduction(reduction)

    def forward(self, y_pred: Tensor, y_true) -> Tensor:
        y_true = _ensure_tensor(y_true)
        if self.reduction == "mean":
            return ops.mae_loss(y_pred, y_true)
        loss = ops.absolute(y_pred - y_true)
        return _reduce(loss, self.reduction)


class L1Loss(MAELoss):
    """Alias for mean absolute error loss."""


class HuberLoss(Module):
    """Huber loss, also known as smooth L1 loss."""

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = _validate_reduction(reduction)

    def forward(self, y_pred: Tensor, y_true) -> Tensor:
        y_true = _ensure_tensor(y_true)
        if self.reduction == "mean":
            return ops.huber_loss(y_pred, y_true, delta=self.delta)
        residual = y_pred - y_true
        abs_residual = ops.absolute(residual)
        quadratic = 0.5 * ops.square(residual)
        linear = self.delta * (abs_residual - 0.5 * self.delta)
        loss = ops.where(abs_residual <= self.delta, quadratic, linear)
        return _reduce(loss, self.reduction)


class SmoothL1Loss(HuberLoss):
    """PyTorch-style smooth L1 wrapper around Huber loss."""

    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__(delta=beta, reduction=reduction)
        self.beta = beta


class LogCoshLoss(Module):
    """Log-cosh regression loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = _validate_reduction(reduction)

    def forward(self, y_pred: Tensor, y_true) -> Tensor:
        y_true = _ensure_tensor(y_true)
        if self.reduction == "mean":
            return ops.log_cosh_loss(y_pred, y_true)
        residual = ops.absolute(y_pred - y_true)
        loss = residual + ops.log1p(ops.exp(-2.0 * residual)) - ops.log(2.0)
        return _reduce(loss, self.reduction)


class NLLLoss(Module):
    """Negative log-likelihood loss from log-probabilities and integer targets."""

    def forward(self, log_probs: Tensor, targets) -> Tensor:
        if isinstance(targets, Tensor):
            targets = targets.detach().to(log_probs.device)
        else:
            targets = to_device(targets, log_probs.device)
        return ops.nll_loss(log_probs, targets)


class KLDivLoss(Module):
    """KL divergence KL(p || q)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = _validate_reduction(reduction)

    def forward(self, p: Tensor, q) -> Tensor:
        q = _ensure_tensor(q)
        if self.reduction == "mean":
            return ops.kl_divergence_loss(p, q)
        eps = safe_eps(p)
        p_safe = ops.clip(p, eps, 1.0)
        q_safe = ops.clip(q, eps, 1.0)
        loss = p_safe * (ops.log(p_safe) - ops.log(q_safe))
        return _reduce(loss, self.reduction)


class FocalLoss(Module):
    """Binary focal loss from logits."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, y_true) -> Tensor:
        y_true = _ensure_tensor(y_true)
        return ops.focal_loss(logits, y_true, gamma=self.gamma, alpha=self.alpha)


class HingeLoss(Module):
    """Binary hinge loss with targets in {-1, +1}."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = _validate_reduction(reduction)

    def forward(self, y_pred: Tensor, y_true) -> Tensor:
        y_true = _ensure_tensor(y_true)
        if self.reduction == "mean":
            return ops.hinge_loss(y_pred, y_true)
        loss = ops.maximum(0.0, 1.0 - y_true * y_pred)
        return _reduce(loss, self.reduction)


class SquaredHingeLoss(Module):
    """Binary squared hinge loss with targets in {-1, +1}."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = _validate_reduction(reduction)

    def forward(self, y_pred: Tensor, y_true) -> Tensor:
        y_true = _ensure_tensor(y_true)
        if self.reduction == "mean":
            return ops.squared_hinge_loss(y_pred, y_true)
        hinge = ops.maximum(0.0, 1.0 - y_true * y_pred)
        loss = ops.square(hinge)
        return _reduce(loss, self.reduction)
