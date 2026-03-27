"""
vjp_rules_test.py
=================
Tests for dnp/core/vjp_rules.py:
  - unbroadcast utility
  - All activation / helper functions (forward values)
  - VJP_RULES correctness for every registered operation
"""

# Third-party
import pytest
import numpy as np

# Local imports
from dnp.core.backend import backend
from dnp.core.vjp_rules import (
    VJP_RULES,
    EPSILON,
    unbroadcast,
    sigmoid,
    relu,
    leaky_relu,
    elu,
    softplus,
    swish,
    gelu,
    softmax,
    _reshape,
    conv2d,
    rot180,
    conv2d_full,
    rope,
    flash_attention,
    sinkhorn,
    neural_ode_solve,
    s4_scan,
)

close = lambda a, b: np.allclose(a, b, atol=1e-5)

# ===========================================================================
# unbroadcast
# ===========================================================================


class TestUnbroadcast:
    def test_same_shape_noop(self):
        g = np.array([1.0, 2.0, 3.0])
        assert close(unbroadcast(g, (3,)), g)

    def test_reduces_added_leading_dims(self):
        g = np.ones((4, 3))  # broadcast from shape (3,)
        result = unbroadcast(g, (3,))
        assert result.shape == (3,)
        assert close(result, np.full((3,), 4.0))

    def test_reduces_keepdim_broadcast(self):
        g = np.ones((3, 4))
        result = unbroadcast(g, (1, 4))
        assert result.shape == (1, 4)


# ===========================================================================
# Activation functions (forward pass)
# ===========================================================================


class TestActivationFunctions:
    def test_sigmoid_zero(self):
        assert close(sigmoid(np.array([0.0])), np.array([0.5]))

    def test_sigmoid_large_positive(self):
        assert close(sigmoid(np.array([600.0])), np.array([1.0]))

    def test_sigmoid_large_negative(self):
        assert close(sigmoid(np.array([-600.0])), np.array([0.0]))

    def test_relu_negative_zeroed(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 3.0])
        assert close(relu(x), expected)

    def test_leaky_relu_negative_slope(self):
        x = np.array([-2.0, 2.0])
        result = leaky_relu(x, alpha=0.1)
        assert close(result, np.array([-0.2, 2.0]))

    def test_elu_positive_passthrough(self):
        x = np.array([1.0, 2.0])
        assert close(elu(x), x)

    def test_elu_negative_saturates(self):
        x = np.array([-100.0])
        assert close(elu(x), np.array([-1.0]))

    def test_softplus_near_zero(self):
        # softplus(0) = log(2) ≈ 0.6931
        assert close(softplus(np.array([0.0])), np.array([np.log(2)]))

    def test_swish_zero(self):
        assert close(swish(np.array([0.0])), np.array([0.0]))

    def test_softmax_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = softmax(x, axis=-1)
        assert close(result.sum(axis=-1), np.ones(1))

    def test_softmax_stable_large_values(self):
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = softmax(x, axis=-1)
        assert np.all(np.isfinite(result))
        assert close(result.sum(axis=-1), np.ones(1))


# ===========================================================================
# Convolution helpers
# ===========================================================================


class TestConvHelpers:
    def test_conv2d_valid_shape(self):
        x = np.random.randn(5, 5)
        w = np.random.randn(3, 3)
        out = conv2d(x, w, mode="valid")
        assert out.shape == (3, 3)

    def test_conv2d_same_shape(self):
        x = np.random.randn(5, 5)
        w = np.random.randn(3, 3)
        out = conv2d(x, w, mode="same")
        assert out.shape == (5, 5)

    def test_conv2d_full_shape(self):
        x = np.random.randn(5, 5)
        w = np.random.randn(3, 3)
        out = conv2d(x, w, mode="full")
        assert out.shape == (7, 7)

    def test_rot180_identity_after_double_rotation(self):
        w = np.arange(9).reshape(3, 3).astype(float)
        assert close(rot180(rot180(w)), w)

    def test_rot180_correctness(self):
        w = np.array([[1, 2], [3, 4]], dtype=float)
        expected = np.array([[4, 3], [2, 1]], dtype=float)
        assert close(rot180(w), expected)

    def test_conv2d_full_helper(self):
        g = np.random.randn(3, 3)
        w = np.random.randn(3, 3)
        out = conv2d_full(g, w)
        assert out.shape == (5, 5)


# ===========================================================================
# VJP rules — correctness checks
# ===========================================================================


class TestVJPBinaryOps:
    x = np.array([2.0, 3.0])
    y = np.array([4.0, 5.0])
    g = np.array([1.0, 1.0])

    def test_add(self):
        gx, gy = VJP_RULES[np.add](self.g, self.x, self.y)
        assert close(gx, self.g) and close(gy, self.g)

    def test_subtract(self):
        gx, gy = VJP_RULES[np.subtract](self.g, self.x, self.y)
        assert close(gx, self.g) and close(gy, -self.g)

    def test_multiply(self):
        gx, gy = VJP_RULES[np.multiply](self.g, self.x, self.y)
        assert close(gx, self.g * self.y) and close(gy, self.g * self.x)

    def test_divide(self):
        gx, gy = VJP_RULES[np.divide](self.g, self.x, self.y)
        assert close(gx, self.g / self.y)
        assert close(gy, -self.g * self.x / self.y**2)

    def test_power(self):
        gx, gy = VJP_RULES[np.power](self.g, self.x, self.y)
        # d/dx x^y = y * x^(y-1)
        assert close(gx, self.g * self.y * np.power(self.x, self.y - 1))

    def test_maximum(self):
        gx, gy = VJP_RULES[np.maximum](self.g, self.x, self.y)
        # x=[2,3], y=[4,5] → y always larger → gx=0, gy=g
        assert close(gx, np.zeros_like(self.x)) and close(gy, self.g)

    def test_minimum(self):
        gx, gy = VJP_RULES[np.minimum](self.g, self.x, self.y)
        # x=[2,3] < y=[4,5] → x is the minimum → gx=g, gy=0
        assert close(gx, self.g) and close(gy, np.zeros_like(self.y))

    def test_matmul(self):
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 5)
        g = np.random.randn(3, 5)
        gA, gB = VJP_RULES[np.matmul](g, A, B)
        assert gA.shape == A.shape and gB.shape == B.shape

    def test_dot(self):
        a = np.random.randn(4)
        b = np.random.randn(4)
        g = np.array(1.0)
        ga, gb = VJP_RULES[np.dot](g, a, b)
        assert ga.shape == a.shape


class TestVJPUnaryOps:
    x = np.array([1.0, 2.0, 3.0])
    g = np.ones(3)

    def test_negative(self):
        (gx,) = VJP_RULES[np.negative](self.g, self.x)
        assert close(gx, -self.g)

    def test_square(self):
        (gx,) = VJP_RULES[np.square](self.g, self.x)
        assert close(gx, 2 * self.x)

    def test_sqrt(self):
        (gx,) = VJP_RULES[np.sqrt](self.g, self.x)
        assert close(gx, 1.0 / (2 * np.sqrt(self.x)))

    def test_exp(self):
        (gx,) = VJP_RULES[np.exp](self.g, self.x)
        assert close(gx, np.exp(self.x))

    def test_log(self):
        (gx,) = VJP_RULES[np.log](self.g, self.x)
        assert close(gx, 1.0 / self.x)

    def test_log1p(self):
        (gx,) = VJP_RULES[np.log1p](self.g, self.x)
        assert close(gx, 1.0 / (self.x + 1.0))

    def test_abs(self):
        x = np.array([-2.0, 3.0])
        (gx,) = VJP_RULES[np.abs](np.ones(2), x)
        assert close(gx, np.sign(x))

    def test_sign_zero_gradient(self):
        (gx,) = VJP_RULES[np.sign](self.g, self.x)
        assert close(gx, np.zeros_like(self.x))

    def test_sin(self):
        (gx,) = VJP_RULES[np.sin](self.g, self.x)
        assert close(gx, np.cos(self.x))

    def test_cos(self):
        (gx,) = VJP_RULES[np.cos](self.g, self.x)
        assert close(gx, -np.sin(self.x))

    def test_tanh(self):
        (gx,) = VJP_RULES[np.tanh](self.g, self.x)
        assert close(gx, 1 - np.tanh(self.x) ** 2)

    def test_floor_zero_gradient(self):
        (gx,) = VJP_RULES[np.floor](self.g, self.x)
        assert close(gx, np.zeros_like(self.x))

    def test_ceil_zero_gradient(self):
        (gx,) = VJP_RULES[np.ceil](self.g, self.x)
        assert close(gx, np.zeros_like(self.x))


class TestVJPReductions:
    def test_sum_broadcasts_gradient(self):
        x = np.array([1.0, 2.0, 3.0])
        g = np.array(2.0)
        (gx,) = VJP_RULES[np.sum](g, x)
        assert close(gx, np.full_like(x, 2.0))

    def test_mean_uniform_gradient(self):
        x = np.array([1.0, 2.0, 4.0])
        g = np.array(1.0)
        (gx,) = VJP_RULES[np.mean](g, x)
        assert close(gx, np.ones_like(x) / 3.0)

    def test_max_indicator(self):
        x = np.array([1.0, 5.0, 3.0])
        g = np.array(1.0)
        (gx,) = VJP_RULES[np.max](g, x)
        expected = np.array([0.0, 1.0, 0.0])
        assert close(gx, expected)

    def test_min_indicator(self):
        x = np.array([1.0, 5.0, 3.0])
        g = np.array(1.0)
        (gx,) = VJP_RULES[np.min](g, x)
        expected = np.array([1.0, 0.0, 0.0])
        assert close(gx, expected)

    def test_prod_gradient(self):
        x = np.array([2.0, 3.0, 4.0])
        g = np.array(1.0)
        (gx,) = VJP_RULES[np.prod](g, x)
        # d/dx_i prod(x) = prod(x) / x_i
        expected = np.array([12.0, 8.0, 6.0])
        assert close(gx, expected)


class TestVJPShapeOps:
    def test_reshape(self):
        x = np.random.randn(3, 4)
        g = np.random.randn(12)
        (gx,) = VJP_RULES[_reshape](g, x, (12,))
        assert gx.shape == (3, 4)

    def test_transpose_2d(self):
        x = np.random.randn(3, 4)
        g = np.random.randn(4, 3)
        (gx,) = VJP_RULES[np.transpose](g, x)
        assert gx.shape == (3, 4)

    def test_expand_dims_then_squeeze(self):
        x = np.random.randn(3, 4)
        g = np.random.randn(1, 3, 4)
        (gx,) = VJP_RULES[np.expand_dims](g, x, 0)
        assert gx.shape == (3, 4)

    def test_squeeze(self):
        x = np.random.randn(1, 4)
        g = np.random.randn(4)
        (gx,) = VJP_RULES[np.squeeze](g, x, axis=0)
        assert gx.shape == (1, 4)


class TestVJPNNOps:
    def test_sigmoid_vjp(self):
        x = np.array([0.0, 1.0])
        g = np.ones(2)
        (gx,) = VJP_RULES[sigmoid](g, x)
        s = sigmoid(x)
        assert close(gx, s * (1 - s))

    def test_relu_vjp_positive(self):
        x = np.array([1.0, -1.0])
        g = np.ones(2)
        (gx,) = VJP_RULES[relu](g, x)
        assert close(gx, np.array([1.0, 0.0]))

    def test_leaky_relu_vjp(self):
        x = np.array([1.0, -1.0])
        g = np.ones(2)
        (gx,) = VJP_RULES[leaky_relu](g, x)
        assert close(gx, np.array([1.0, 0.01]))

    def test_elu_vjp(self):
        x = np.array([1.0, -1.0])
        g = np.ones(2)
        (gx,) = VJP_RULES[elu](g, x)
        expected = np.where(x > 0, 1.0, np.exp(x))
        assert close(gx, expected)

    def test_softplus_vjp(self):
        x = np.array([0.0, 1.0])
        g = np.ones(2)
        (gx,) = VJP_RULES[softplus](g, x)
        assert close(gx, sigmoid(x))

    def test_softmax_vjp_shape(self):
        x = np.array([1.0, 2.0, 3.0])
        g = np.ones(3)
        (gx,) = VJP_RULES[softmax](g, x)
        assert gx.shape == x.shape

    def test_conv2d_vjp_valid_shapes(self):
        x = np.random.randn(5, 5)
        w = np.random.randn(3, 3)
        g = np.random.randn(3, 3)
        gx, gw = VJP_RULES[conv2d](g, x, w, mode="valid")
        assert gx.shape == (5, 5)
        assert gw.shape == (3, 3)

    def test_conv2d_vjp_same_shapes(self):
        x = np.random.randn(5, 5)
        w = np.random.randn(3, 3)
        g = np.random.randn(5, 5)
        gx, gw = VJP_RULES[conv2d](g, x, w, mode="same")
        assert gx.shape == (5, 5)
        assert gw.shape == (3, 3)

    def test_conv2d_vjp_full_shapes(self):
        x = np.random.randn(5, 5)
        w = np.random.randn(3, 3)
        g = np.random.randn(7, 7)
        gx, gw = VJP_RULES[conv2d](g, x, w, mode="full")
        assert gx.shape == (5, 5)
        assert gw.shape == (3, 3)


class TestVJPAdvancedOps:
    def test_rope_vjp_shape_and_reversibility(self):
        batch, seq_len, dim = 2, 5, 8
        x = np.random.randn(batch, seq_len, dim)
        cos_freqs = np.random.randn(batch, seq_len, dim // 2)
        sin_freqs = np.random.randn(batch, seq_len, dim // 2)
        g = np.random.randn(batch, seq_len, dim)
        
        gx, gcos, gsin = VJP_RULES[rope](g, x, cos_freqs, sin_freqs)
        
        assert gx.shape == x.shape
        assert gcos is None
        assert gsin is None
        # Forward rope with positive angle, backward is negative angle.
        # Roping g with negative angle should equal gx.
        fwd_rope = rope(g, cos_freqs, -sin_freqs)
        assert close(gx, fwd_rope)

    def test_flash_attention_vjp_shapes(self):
        B, N, M, d_k, d_v = 2, 5, 5, 4, 8
        Q = np.random.randn(B, N, d_k)
        K = np.random.randn(B, M, d_k)
        V = np.random.randn(B, M, d_v)
        g_out = np.random.randn(B, N, d_v)
        
        dQ, dK, dV, _ = VJP_RULES[flash_attention](g_out, Q, K, V, mask=None)
        
        assert dQ.shape == Q.shape
        assert dK.shape == K.shape
        assert dV.shape == V.shape

    def test_sinkhorn_vjp_shapes(self):
        B, N, M = 2, 10, 10
        a = np.ones((B, N)) / N
        b = np.ones((B, M)) / M
        cost_M = np.random.randn(B, N, M)
        g_P = np.random.randn(B, N, M)
        reg = 0.1
        
        da, db, dM, dreg, _ = VJP_RULES[sinkhorn](g_P, a, b, cost_M, reg, 5)
        
        assert da is None
        assert db is None
        assert dM.shape == cost_M.shape
        
    def test_neural_ode_vjp_shapes(self):
        z0 = np.random.randn(3, 4)
        t_span = (0.0, 1.0)
        g_z = np.random.randn(3, 4)
        
        dz0, dt, dstep = VJP_RULES[neural_ode_solve](g_z, z0, t_span, 5)
        
        assert dz0.shape == z0.shape
        assert dt is None
        
    def test_s4_scan_vjp_shapes(self):
        batch, seq_len, d_in, d_model, d_out = 2, 5, 3, 4, 6
        u = np.random.randn(batch, seq_len, d_in)
        A = np.random.randn(d_model, d_model)
        B = np.random.randn(d_in, d_model)
        C = np.random.randn(d_model, d_out)
        g_y = np.random.randn(batch, seq_len, d_out)
        
        du, dA, dB, dC = VJP_RULES[s4_scan](g_y, u, A, B, C)
        
        assert du.shape == u.shape
        assert dB.shape == B.shape
        assert dC.shape == C.shape

    def test_fft_vjp_shapes_and_values(self):
        # 1D complex input
        x = np.random.randn(5) + 1j * np.random.randn(5)
        g = np.random.randn(5) + 1j * np.random.randn(5)
        n = 5
        
        gx, = VJP_RULES[backend.scipy.fft.fft](g, x)
        assert gx.shape == x.shape
        # Check vjp(fft) == n * ifft(g)
        assert close(gx, n * np.fft.ifft(g))

    def test_ifft_vjp_shapes_and_values(self):
        x = np.random.randn(4) + 1j * np.random.randn(4)
        g = np.random.randn(4) + 1j * np.random.randn(4)
        n = 4
        
        gx, = VJP_RULES[backend.scipy.fft.ifft](g, x)
        assert gx.shape == x.shape
        assert close(gx, (1.0 / n) * np.fft.fft(g))

    def test_fftn_vjp_shapes(self):
        x = np.random.randn(3, 4) + 1j * np.random.randn(3, 4)
        g = np.random.randn(3, 4) + 1j * np.random.randn(3, 4)
        n = 12
        
        gx, = VJP_RULES[backend.scipy.fft.fftn](g, x)
        assert gx.shape == x.shape
        assert close(gx, n * np.fft.ifftn(g))

    def test_ifftn_vjp_shapes(self):
        x = np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)
        g = np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)
        n = 8
        
        gx, = VJP_RULES[backend.scipy.fft.ifftn](g, x)
        assert gx.shape == x.shape
        assert close(gx, (1.0 / n) * np.fft.fftn(g))

