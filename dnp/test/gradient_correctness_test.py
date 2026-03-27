"""
gradient_correctness_test.py
=============================
Finite-difference gradient correctness checks (T1, T2, T6) and an
end-to-end training integration test (T10).

Each VJP is verified by comparing the autograd gradient against a
central-difference numerical estimate:
    g_fd[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
"""

import pytest
import numpy as np

from dnp.core.tensor import Tensor
from dnp.core import ops
from dnp.core.session import session
import dnp.core.vjp_rules as rules


# ---------------------------------------------------------------------------
# Finite-difference helper
# ---------------------------------------------------------------------------

EPS = 1e-4  # sufficiently large step size for float64


def _fd_grad(scalar_fn, x_np, eps=EPS):
    """Central-difference gradient of scalar_fn w.r.t. x_np (flat loop)."""
    grad = np.zeros_like(x_np, dtype=np.float64)
    x_flat = x_np.ravel()
    for i in range(x_flat.size):
        x_p = x_flat.copy()
        x_p[i] += eps
        x_m = x_flat.copy()
        x_m[i] -= eps
        fp = float(scalar_fn(x_p.reshape(x_np.shape)))
        fm = float(scalar_fn(x_m.reshape(x_np.shape)))
        grad.flat[i] = (fp - fm) / (2 * eps)
    return grad


def _autograd_grad(op_fn, x_np, y_np=None):
    """Run op_fn(x[, y]) → scalar, backward, return x.grad as numpy array."""
    session.reset()
    x = Tensor(x_np.astype(np.float64))
    with session.graph():
        result = op_fn(x, y_np) if y_np is not None else op_fn(x)
        # Sum to scalar if needed
        if result.data.ndim > 0:
            result = ops.sum(result)
    result.backward()
    return np.array(x.grad, dtype=np.float64)


# ===========================================================================
# T1 — Loss function VJP correctness (finite-difference checks)
# ===========================================================================


class TestLossVJPs:
    """Verify autograd gradients for loss functions against finite differences.

    Coverage: mse, mae, huber, bce, bce_with_logits, cce_with_logits,
              hinge, squared_hinge, kl_divergence.
    """

    tol = 1e-4  # absolute tolerance for gradient comparison

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _loss_grad_pred(op_name, pred, target, **kwargs):
        """Return autograd gradient of ops.<op_name>(pred, target) w.r.t. pred."""
        session.reset()
        pred_t = Tensor(pred.astype(np.float64))
        target_t = Tensor(target.astype(np.float64), requires_grad=False)
        loss_op = getattr(ops, op_name)
        with session.graph():
            loss = loss_op(pred_t, target_t, **kwargs)
            if isinstance(loss, Tensor) and loss.data.ndim > 0:
                loss = ops.sum(loss)
        loss.backward()
        return np.array(pred_t.grad, dtype=np.float64)

    @staticmethod
    def _loss_fd_grad(op_name, pred, target, **kwargs):
        """Central-difference gradient of ops.<op_name> w.r.t. pred (numerical)."""
        loss_fn_np = getattr(rules, op_name)

        def scalar_fn(p):
            v = loss_fn_np(p, target, **kwargs)
            return float(v.sum() if hasattr(v, "sum") else v)

        return _fd_grad(scalar_fn, pred)

    # ── MSE ─────────────────────────────────────────────────────────────────

    def test_mse_loss_vjp(self):
        pred = np.array([0.5, 1.2, -0.3, 0.8])
        target = np.array([1.0, 0.5, 0.0, 1.0])
        auto = self._loss_grad_pred("mse_loss", pred, target)
        fd = self._loss_fd_grad("mse_loss", pred, target)
        assert np.allclose(auto, fd, atol=self.tol), f"MSE VJP mismatch: {auto} vs {fd}"

    # ── MAE ─────────────────────────────────────────────────────────────────

    def test_mae_loss_vjp(self):
        pred = np.array([0.5, 1.2, -0.3, 0.8])
        target = np.array([1.0, 0.5, 0.1, 1.0])  # slight offset avoids kink at 0
        auto = self._loss_grad_pred("mae_loss", pred, target)
        fd = self._loss_fd_grad("mae_loss", pred, target)
        assert np.allclose(auto, fd, atol=self.tol), f"MAE VJP mismatch"

    # ── Huber ────────────────────────────────────────────────────────────────

    def test_huber_loss_vjp(self):
        pred = np.array([0.5, 2.0, -0.3])
        target = np.array([0.0, 0.0, 0.0])
        auto = self._loss_grad_pred("huber_loss", pred, target)
        fd = self._loss_fd_grad("huber_loss", pred, target)
        assert np.allclose(auto, fd, atol=self.tol)

    # ── BCE ──────────────────────────────────────────────────────────────────

    def test_bce_loss_vjp(self):
        pred = np.array([0.7, 0.2, 0.9, 0.4])  # probabilities in (0,1)
        target = np.array([1.0, 0.0, 1.0, 0.0])
        auto = self._loss_grad_pred("bce_loss", pred, target)
        fd = self._loss_fd_grad("bce_loss", pred, target)
        assert np.allclose(auto, fd, atol=self.tol)

    # ── BCE with logits ───────────────────────────────────────────────────────

    def test_bce_with_logits_vjp(self):
        pred = np.array([1.0, -0.5, 2.0, -1.0])  # raw logits
        target = np.array([1.0, 0.0, 1.0, 0.0])
        auto = self._loss_grad_pred("bce_with_logits_loss", pred, target)
        fd = self._loss_fd_grad("bce_with_logits_loss", pred, target)
        assert np.allclose(auto, fd, atol=self.tol)

    # ── CCE with logits ───────────────────────────────────────────────────────

    def test_cce_with_logits_vjp(self):
        pred = np.array([[1.0, 2.0, 0.5], [0.3, -0.1, 1.5]])  # (B, C) logits
        target = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # one-hot (B, C)
        auto = self._loss_grad_pred("cce_with_logits_loss", pred, target)
        fd = self._loss_fd_grad("cce_with_logits_loss", pred, target)
        assert np.allclose(auto, fd, atol=self.tol)

    # ── Hinge ────────────────────────────────────────────────────────────────

    def test_hinge_loss_vjp(self):
        pred = np.array([0.8, -0.3, 1.2])  # predictions
        target = np.array([1.0, -1.0, 1.0])  # +1/-1 labels
        auto = self._loss_grad_pred("hinge_loss", pred, target)
        fd = self._loss_fd_grad("hinge_loss", pred, target)
        assert np.allclose(auto, fd, atol=self.tol)

    # ── Squared hinge ────────────────────────────────────────────────────────

    def test_squared_hinge_loss_vjp(self):
        pred = np.array([0.8, -0.3, 1.2])
        target = np.array([1.0, -1.0, 1.0])
        auto = self._loss_grad_pred("squared_hinge_loss", pred, target)
        fd = self._loss_fd_grad("squared_hinge_loss", pred, target)
        assert np.allclose(auto, fd, atol=self.tol)

    # ── KL Divergence ────────────────────────────────────────────────────────

    def test_kl_divergence_vjp(self):
        pred = np.array([0.3, 0.4, 0.3])  # predicted distribution
        target = np.array([0.2, 0.5, 0.3])
        auto = self._loss_grad_pred("kl_divergence_loss", pred, target)
        fd = self._loss_fd_grad("kl_divergence_loss", pred, target)
        assert np.allclose(auto, fd, atol=self.tol)

    # ── Log-cosh ─────────────────────────────────────────────────────────────

    def test_log_cosh_loss_vjp(self):
        pred = np.array([0.5, 1.5, -0.4])
        target = np.array([0.0, 1.0, 0.0])
        auto = self._loss_grad_pred("log_cosh_loss", pred, target)
        fd = self._loss_fd_grad("log_cosh_loss", pred, target)
        assert np.allclose(auto, fd, atol=self.tol)


# ===========================================================================
# T2 — RNN / LSTM / GRU gradient correctness
# ===========================================================================


class TestRNNGradients:
    """Check that rnn_cell / lstm_cell / gru_cell VJPs match finite differences."""

    tol = 1e-3
    B, T, d_in, d_h = 1, 3, 4, 5  # small dims for speed

    def _rnn_loss(self, x_np, Wh_np, Wx_np, bh_np):
        h_seq = rules.rnn_cell(x_np, Wh_np, Wx_np, bh_np)
        return h_seq.sum()

    def _lstm_loss(self, x_np, W_np, b_np):
        h_seq = rules.lstm_cell(x_np, W_np, b_np)
        return h_seq.sum()

    def _gru_loss(self, x_np, Wr_np, Wz_np, Wh_np, br_np, bz_np, bh_np):
        h_seq = rules.gru_cell(x_np, Wr_np, Wz_np, Wh_np, br_np, bz_np, bh_np)
        return h_seq.sum()

    def test_rnn_input_grad(self):
        np.random.seed(0)
        B, T, d_in, d_h = self.B, self.T, self.d_in, self.d_h
        x = np.random.randn(B, T, d_in)
        Wh = np.random.randn(d_h, d_h) * 0.1
        Wx = np.random.randn(d_in, d_h) * 0.1
        bh = np.zeros(d_h)

        # Autograd
        session.reset()
        x_t = Tensor(x)
        Wh_t = Tensor(Wh)
        Wx_t = Tensor(Wx)
        bh_t = Tensor(bh)
        with session.graph():
            out = ops.rnn_cell(x_t, Wh_t, Wx_t, bh_t)
            ops.sum(out).backward()
        auto = np.array(x_t.grad)

        # Finite difference
        fd = _fd_grad(lambda xp: self._rnn_loss(xp, Wh, Wx, bh), x)
        assert np.allclose(auto, fd, atol=self.tol), "RNN input gradient mismatch"

    def test_rnn_weight_grad(self):
        np.random.seed(1)
        B, T, d_in, d_h = self.B, self.T, self.d_in, self.d_h
        x = np.random.randn(B, T, d_in)
        Wh = np.random.randn(d_h, d_h) * 0.1
        Wx = np.random.randn(d_in, d_h) * 0.1
        bh = np.zeros(d_h)

        session.reset()
        x_t = Tensor(x)
        Wh_t = Tensor(Wh)
        Wx_t = Tensor(Wx)
        bh_t = Tensor(bh)
        with session.graph():
            out = ops.rnn_cell(x_t, Wh_t, Wx_t, bh_t)
            ops.sum(out).backward()
        auto = np.array(Wx_t.grad)
        fd = _fd_grad(lambda Wxa: self._rnn_loss(x, Wh, Wxa, bh), Wx)
        assert np.allclose(auto, fd, atol=self.tol), "RNN Wx gradient mismatch"

    def test_lstm_input_grad(self):
        np.random.seed(2)
        B, T, d_in, d_h = self.B, self.T, self.d_in, self.d_h
        x = np.random.randn(B, T, d_in)
        W = np.random.randn(d_in + d_h, 4 * d_h) * 0.1
        b = np.zeros(4 * d_h)

        session.reset()
        x_t = Tensor(x)
        W_t = Tensor(W)
        b_t = Tensor(b)
        with session.graph():
            out = ops.lstm_cell(x_t, W_t, b_t)
            ops.sum(out).backward()
        auto = np.array(x_t.grad)
        fd = _fd_grad(lambda xp: self._lstm_loss(xp, W, b), x)
        assert np.allclose(auto, fd, atol=self.tol), "LSTM input gradient mismatch"

    def test_gru_input_grad(self):
        np.random.seed(3)
        B, T, d_in, d_h = self.B, self.T, self.d_in, self.d_h
        x = np.random.randn(B, T, d_in)
        d = d_in + d_h
        Wr = np.random.randn(d, d_h) * 0.1
        Wz = np.random.randn(d, d_h) * 0.1
        Wh_g = np.random.randn(d, d_h) * 0.1
        br = np.zeros(d_h)
        bz = np.zeros(d_h)
        bh = np.zeros(d_h)

        session.reset()
        x_t = Tensor(x)
        Wr_t = Tensor(Wr)
        Wz_t = Tensor(Wz)
        Wh_t = Tensor(Wh_g)
        br_t = Tensor(br)
        bz_t = Tensor(bz)
        bh_t = Tensor(bh)
        with session.graph():
            out = ops.gru_cell(x_t, Wr_t, Wz_t, Wh_t, br_t, bz_t, bh_t)
            ops.sum(out).backward()
        auto = np.array(x_t.grad)
        fd = _fd_grad(lambda xp: self._gru_loss(xp, Wr, Wz, Wh_g, br, bz, bh), x)
        assert np.allclose(auto, fd, atol=self.tol), "GRU input gradient mismatch"


# ===========================================================================
# T6 — Conv1d gradient correctness
# ===========================================================================


class TestConv1dGrad:
    """Verify that Conv1d layer gradients match finite differences."""

    tol = 1e-3

    def test_conv1d_input_grad(self):
        from dnp.layers import Conv1d

        np.random.seed(7)
        layer = Conv1d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
        x_np = np.random.randn(1, 2, 8).astype(np.float64)

        # Autograd
        session.reset()
        x_t = Tensor(x_np)
        layer.zero_grad()
        with session.graph():
            out = layer(x_t)
            ops.sum(out).backward()
        auto = np.array(x_t.grad)

        # Finite difference
        def fwd_np(xp):
            session.reset()
            xt = Tensor(xp)
            with session.graph():
                o = layer(xt)
            return float(o.data.sum())

        fd = _fd_grad(fwd_np, x_np)
        assert np.allclose(auto, fd, atol=self.tol), "Conv1d input gradient mismatch"

    def test_conv1d_weight_grad(self):
        from dnp.layers import Conv1d

        np.random.seed(8)
        layer = Conv1d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
        x_np = np.random.randn(1, 2, 8).astype(np.float64)
        x_t = Tensor(x_np)

        # Autograd
        session.reset()
        layer.zero_grad()
        x_t = Tensor(x_np)
        with session.graph():
            out = layer(x_t)
            ops.sum(out).backward()
        # Use the first weight parameter
        w_param = next(iter(layer.parameters()))
        auto = np.array(w_param.grad)

        # Finite difference
        w_np = w_param.data.copy()

        def fwd_w(wp):
            w_param.data[...] = wp
            session.reset()
            xt = Tensor(x_np)
            with session.graph():
                o = layer(xt)
            return float(o.data.sum())

        fd = _fd_grad(fwd_w, w_np)
        w_param.data[...] = w_np  # restore
        assert np.allclose(auto, fd, atol=self.tol), "Conv1d weight gradient mismatch"


# ===========================================================================
# T3 — Tensor.__getitem__ gradient correctness (B7)
# ===========================================================================


class TestGetitemGrad:
    """Verify that __getitem__ (slice/fancy) gradients are correct."""

    tol = 1e-5

    def test_slice_grad(self):
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        session.reset()
        x_t = Tensor(x_np)
        with session.graph():
            y = x_t[1:4]
            ops.sum(y).backward()
        expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        assert np.allclose(x_t.grad, expected, atol=self.tol)

    def test_fancy_index_grad(self):
        x_np = np.array([1.0, 2.0, 3.0, 4.0])
        idx = np.array([0, 2, 3])
        session.reset()
        x_t = Tensor(x_np)
        with session.graph():
            y = x_t[idx]
            ops.sum(y).backward()
        expected = np.array([1.0, 0.0, 1.0, 1.0])
        assert np.allclose(x_t.grad, expected, atol=self.tol)

    def test_repeated_index_accumulates(self):
        x_np = np.array([1.0, 2.0, 3.0])
        idx = np.array([0, 0, 1])  # index 0 repeated — grad should accumulate
        session.reset()
        x_t = Tensor(x_np)
        with session.graph():
            y = x_t[idx]
            ops.sum(y).backward()
        expected = np.array([2.0, 1.0, 0.0])
        assert np.allclose(x_t.grad, expected, atol=self.tol)


# ===========================================================================
# T10 — XOR integration test: model converges in <100 epochs
# ===========================================================================


class TestXORIntegration:
    """End-to-end training: a 2-layer MLP solves XOR in ≤100 epochs."""

    def test_xor_converges(self):
        import dnp

        np.random.seed(42)

        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([[0.0], [1.0], [1.0], [0.0]])

        model = dnp.Sequential(
            dnp.Linear(2, 8),
            dnp.ReLU(),
            dnp.Linear(8, 1),
            dnp.Sigmoid(),
        )
        optimizer = dnp.Adam(model.parameters(), lr=0.05)

        final_loss = float("inf")
        for epoch in range(100):
            session.reset()
            model.zero_grad()
            x_t = Tensor(X)
            y_t = Tensor(y)
            with session.graph():
                pred = model(x_t)
                loss = ops.mean(
                    ops.negative(
                        y_t * ops.log(pred + 1e-7)
                        + (1.0 - y_t) * ops.log(1.0 - pred + 1e-7)
                    )
                )
            loss.backward()
            optimizer.step()
            final_loss = float(loss.data)

        assert final_loss < 0.1, (
            f"XOR model did not converge in 100 epochs; final loss = {final_loss:.4f}"
        )

        # Verify predictions are correct
        session.reset()
        x_t = Tensor(X)
        with session.graph():
            pred = model(x_t)
        preds = (np.array(pred.data) > 0.5).astype(float)
        assert np.array_equal(preds, y), f"XOR predictions wrong: {preds}"
