"""
Extra tests covering improvement.md items:
  T4  — attention / transformer layer shape + gradient smoke tests
  T5  — where, gather, embedding forward + backward
  T7  — optimizer state management (Adam moments, SGD momentum, StepLR)
  T8  — CPU↔GPU device-switch (skipped when CuPy is absent)
  T9  — BatchNorm eval-mode behaviour (running stats used)
"""

import numpy as np
import pytest

from dnp.core import session
from dnp.core.tensor import Tensor
from dnp.core import ops
from dnp.core.optimizers import Adam, SGD, StepLR
from dnp.layers import (
    MultiHeadAttention,
    TransformerEncoderLayer,
    PositionalEncoding,
    BatchNorm2d,
    Module,
    Linear,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed=0):
    return np.random.default_rng(seed)


# ===========================================================================
# T4 — Attention / Transformer layers
# ===========================================================================


class TestAttentionLayers:
    """Shape outputs and gradient smoke-tests for attention layers."""

    def test_multihead_attention_output_shape(self):
        rng = _rng()
        B, T, D, H = 2, 5, 16, 4
        mha = MultiHeadAttention(d_model=D, num_heads=H)
        x = Tensor(rng.standard_normal((B, T, D)).astype(np.float64))
        session.reset()
        with session.graph():
            out = mha(x, x, x)
        assert out.shape == (B, T, D), f"Expected ({B},{T},{D}), got {out.shape}"

    def test_multihead_attention_backward(self):
        rng = _rng(1)
        B, T, D, H = 2, 4, 8, 2
        mha = MultiHeadAttention(d_model=D, num_heads=H)
        x = Tensor(rng.standard_normal((B, T, D)).astype(np.float64))
        session.reset()
        with session.graph():
            out = mha(x, x, x)
            loss = ops.mean(out)
        loss.backward()
        for param in mha.parameters():
            assert param.grad is not None, f"Param {id(param)} has no grad"

    def test_positional_encoding_shape(self):
        d_model, max_len = 16, 20
        pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        rng = _rng(2)
        x = Tensor(rng.standard_normal((3, 10, d_model)).astype(np.float64))
        session.reset()
        with session.graph():
            out = pe(x)
        assert out.shape == (3, 10, d_model)

    def test_transformer_encoder_layer_shape(self):
        rng = _rng(3)
        B, T, D = 2, 6, 16
        enc = TransformerEncoderLayer(d_model=D, num_heads=4, d_ff=32)
        x = Tensor(rng.standard_normal((B, T, D)).astype(np.float64))
        session.reset()
        with session.graph():
            out = enc(x)
        assert out.shape == (B, T, D)

    def test_transformer_encoder_layer_backward(self):
        rng = _rng(4)
        B, T, D = 2, 4, 8
        enc = TransformerEncoderLayer(d_model=D, num_heads=2, d_ff=16)
        x = Tensor(rng.standard_normal((B, T, D)).astype(np.float64))
        session.reset()
        with session.graph():
            out = enc(x)
            loss = ops.mean(out)
        loss.backward()
        all_grads = [p.grad for p in enc.parameters()]
        assert any(g is not None for g in all_grads), "No grad flowed through encoder"


# ===========================================================================
# T5 — where / gather / embedding
# ===========================================================================


class TestSpecialOps:
    """Forward + backward for where, gather, and embedding kernel-call ops."""

    # --- where ---

    def test_where_output(self):
        cond = np.array([True, False, True, False])
        x = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        y = Tensor(np.array([10.0, 20.0, 30.0, 40.0]))
        session.reset()
        with session.graph():
            out = ops.where(cond, x, y)
        expected = np.array([1.0, 20.0, 3.0, 40.0])
        np.testing.assert_array_equal(np.array(out), expected)

    def test_where_backward_x(self):
        cond = np.array([True, False, True])
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        y = Tensor(np.array([4.0, 5.0, 6.0]))
        session.reset()
        with session.graph():
            out = ops.sum(ops.where(cond, x, y))
        out.backward()
        # grad through x where cond=True, 0 where False
        np.testing.assert_array_equal(np.array(x.grad), [1.0, 0.0, 1.0])

    def test_where_backward_y(self):
        cond = np.array([True, False, True])
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        y = Tensor(np.array([4.0, 5.0, 6.0]))
        session.reset()
        with session.graph():
            out = ops.sum(ops.where(cond, x, y))
        out.backward()
        np.testing.assert_array_equal(np.array(y.grad), [0.0, 1.0, 0.0])

    # --- gather ---

    def test_gather_forward(self):
        params = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        idx = np.array([0, 2, 1])
        session.reset()
        with session.graph():
            out = ops.gather(params, idx)
        expected = np.array([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]])
        np.testing.assert_array_equal(np.array(out), expected)

    def test_gather_backward(self):
        params = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        idx = np.array([0, 2])
        session.reset()
        with session.graph():
            out = ops.gather(params, idx)
            loss = ops.sum(out)
        loss.backward()
        # Row 1 is unselected → zero grad
        assert params.grad is not None
        np.testing.assert_array_equal(np.array(params.grad)[1], [0.0, 0.0])

    # --- embedding ---

    def test_embedding_forward_shape(self):
        vocab_size, embed_dim = 10, 4
        weights = Tensor(np.random.randn(vocab_size, embed_dim).astype(np.float64))
        idx = np.array([0, 3, 7])
        session.reset()
        with session.graph():
            out = ops.embedding(weights, idx)
        assert out.shape == (3, embed_dim)

    def test_embedding_backward(self):
        vocab_size, embed_dim = 8, 3
        weights = Tensor(np.ones((vocab_size, embed_dim), dtype=np.float64))
        idx = np.array([1, 1, 5])  # row 1 repeated → should accumulate
        session.reset()
        with session.graph():
            out = ops.embedding(weights, idx)
            loss = ops.sum(out)
        loss.backward()
        assert weights.grad is not None
        # Row 1 was looked-up twice → grad = 2 * ones per dim
        np.testing.assert_array_equal(np.array(weights.grad)[1], [2.0] * embed_dim)
        # Row 5 looked-up once → grad = ones
        np.testing.assert_array_equal(np.array(weights.grad)[5], [1.0] * embed_dim)
        # Unselected rows → zero grad
        np.testing.assert_array_equal(np.array(weights.grad)[0], [0.0] * embed_dim)


# ===========================================================================
# T7 — Optimizer state management
# ===========================================================================


class TestOptimizerState:
    """Verify Adam moment estimates and SGD momentum update correctly."""

    def _scalar_loss(self, params):
        """Simple L2 loss so grad = 2*param."""
        session.reset()
        with session.graph():
            loss = ops.sum(ops.square(params[0]))
        loss.backward()

    def test_adam_moment_updates(self):
        p = Tensor(np.array([1.0, -1.0]))
        opt = Adam([p], lr=0.001, beta1=0.9, beta2=0.999)

        self._scalar_loss([p])
        opt.step()
        opt.zero_grad()

        m1 = np.array(opt.m[0]).copy()
        v1 = np.array(opt.v[0]).copy()

        assert opt.t == 1
        # m = (1-0.9) * grad = 0.1 * 2*[1,-1]
        np.testing.assert_allclose(m1, 0.1 * 2 * np.array([1.0, -1.0]), atol=1e-10)
        # v = (1-0.999) * grad^2 = 0.001 * 4
        np.testing.assert_allclose(v1, 0.001 * 4 * np.ones(2), atol=1e-10)

    def test_adam_t_increments(self):
        p = Tensor(np.array([1.0]))
        opt = Adam([p], lr=0.01)
        for _ in range(5):
            self._scalar_loss([p])
            opt.step()
            opt.zero_grad()
        assert opt.t == 5

    def test_sgd_applies_grad(self):
        p = Tensor(np.array([3.0, -2.0]))
        opt = SGD([p], lr=0.1)
        self._scalar_loss([p])  # grad = 2*p = [6, -4]
        opt.step()
        np.testing.assert_allclose(
            np.array(p), [3.0 - 0.1 * 6, -2.0 - 0.1 * (-4)], atol=1e-10
        )

    def test_steplr_decays_lr(self):
        p = Tensor(np.array([1.0]))
        opt = SGD([p], lr=1.0)
        scheduler = StepLR(opt, step_size=3, gamma=0.1)
        for step in range(1, 7):
            scheduler.step()
        # After 6 steps with step_size=3, lr decayed twice: 1.0 * 0.1^2 = 0.01
        assert abs(opt.lr - 0.01) < 1e-10, f"Expected lr=0.01, got {opt.lr}"


# ===========================================================================
# T8 — CPU ↔ GPU device switching (skipped if CuPy absent)
# ===========================================================================

try:
    import cupy  # noqa — just checking availability

    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


@pytest.mark.skipif(not _HAS_GPU, reason="CuPy not available")
class TestDeviceSwitching:
    """Tests only run when a GPU/CuPy is available."""

    def test_tensor_to_gpu_and_back(self):
        from dnp.core.backend import set_device

        t = Tensor(np.array([1.0, 2.0, 3.0]))
        t.cuda()
        import cupy as cp

        assert isinstance(t.data, cp.ndarray), "Expected cupy array after .cuda()"
        t.cpu()
        assert isinstance(t.data, np.ndarray), "Expected numpy array after .cpu()"

    def test_grad_stays_on_same_device(self):
        import cupy as cp
        from dnp.core.backend import set_device

        t = Tensor(np.ones((2, 2), dtype=np.float64))
        t.cuda()
        session.reset()
        with session.graph():
            loss = ops.sum(ops.square(t))
        loss.backward()
        assert isinstance(t.grad, cp.ndarray), "Gradient should be on GPU"


# ===========================================================================
# T9 — BatchNorm eval-mode behaviour
# ===========================================================================


class TestBatchNormEval:
    """BatchNorm2d should use running stats in eval mode, not batch stats."""

    def test_eval_uses_running_stats(self):
        rng = _rng(7)
        C = 4
        bn = BatchNorm2d(
            num_features=C, momentum=1.0
        )  # momentum=1 → running = last batch

        # Train pass: flush batch stats into running stats
        x_train = Tensor(rng.standard_normal((8, C, 4, 4)).astype(np.float64))
        session.reset()
        with session.graph():
            _ = bn(x_train)

        running_mean = np.array(bn.running_mean).copy()
        running_var = np.array(bn.running_var).copy()

        # Switch to eval
        bn.eval()
        assert not bn.training, "Model should be in eval mode"

        # eval pass with very different statistics
        x_eval = Tensor(
            (rng.standard_normal((2, C, 4, 4)) * 100.0 + 50.0).astype(np.float64)
        )
        session.reset()
        with session.graph():
            out_eval = bn(x_eval)

        # Reconstruct expected output using running stats
        eps = bn.eps
        scale = np.array(bn.weight).reshape(1, C, 1, 1)
        bias_ = np.array(bn.bias).reshape(1, C, 1, 1)
        m = running_mean.reshape(1, C, 1, 1)
        v = running_var.reshape(1, C, 1, 1)
        expected = ((np.array(x_eval) - m) / np.sqrt(v + eps)) * scale + bias_

        np.testing.assert_allclose(np.array(out_eval), expected, atol=1e-6)

    def test_train_eval_toggle(self):
        bn = BatchNorm2d(num_features=3)
        assert bn.training, "Default should be training=True"
        bn.eval()
        assert not bn.training, "After eval(), training should be False"
        bn.train()
        assert bn.training, "After train(), training should be True"
