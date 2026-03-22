"""dnp/test/multibranch_test.py
==============================
Tests for reverse-mode autodiff on graphs with multiple branches:

  Diamond          — one node fans out to two ops whose outputs are combined
  Shared weights   — same parameter used in two independent computations
  Multi-use leaf   — leaf appears in >2 downstream nodes
  Deep chain       — wide depth to verify topo-sort correctness
  Self-product     — x * x (x used twice in a single op)
  Quadratic        — f(x) = (ax + b)^2 compared against finite-difference
  MLP loss         — end-to-end forward+backward through a small dense network
  Gradient of sum  — loss = sum(x + y) + sum(x * y), gradients computed analytically
  Chained splits   — output used in three different branches then recombined
  Scalar broadcast — scalar tensor broadcast across a matrix op
"""

import numpy as np
import pytest

import dnp.core.ops as ops
from dnp.core.tensor import Tensor
from dnp.core.session import session as _global_session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_atol = 1e-5
close = lambda a, b: np.allclose(np.asarray(a), np.asarray(b), atol=_atol)


def _T(data, name="x"):
    return Tensor(np.asarray(data, dtype=np.float64), name=name)


def _reset_grads(*tensors):
    for t in tensors:
        t.grad[:] = 0.0


def _finite_diff(fn, x_np, idx, eps=1e-5):
    """Numerical gradient for a scalar-output function at index `idx`."""
    x_plus = x_np.copy()
    x_plus.flat[idx] += eps
    x_minus = x_np.copy()
    x_minus.flat[idx] -= eps
    return (fn(x_plus) - fn(x_minus)) / (2 * eps)


# ---------------------------------------------------------------------------
# Diamond graph
#
#       x
#      / \
#     a   b         a = sin(x),  b = cos(x)
#      \ /
#       y            y = a + b   → scalar loss = sum(y)
#
# d(loss)/dx = cos(x) - sin(x)
# ---------------------------------------------------------------------------


class TestDiamondGraph:
    def test_diamond_gradient(self):
        x = _T(np.array([0.0, np.pi / 2, np.pi]))
        a = ops.sin(x)
        b = ops.cos(x)
        y = ops.add(a, b)
        loss = ops.sum(y)
        loss.backward()

        expected = np.cos(np.array([0.0, np.pi / 2, np.pi])) - np.sin(
            np.array([0.0, np.pi / 2, np.pi])
        )
        assert close(x.grad, expected), f"diamond grad mismatch: {x.grad} vs {expected}"

    def test_diamond_grad_accumulates_not_overwrites(self):
        """x.grad must be the SUM of both branches, not the last one written."""
        x = _T([1.0, 2.0])
        a = ops.exp(x)  # da/dx = exp(x)
        b = ops.square(x)  # db/dx = 2x
        loss = ops.sum(ops.add(a, b))
        loss.backward()

        x_np = np.array([1.0, 2.0])
        expected = np.exp(x_np) + 2 * x_np
        assert close(x.grad, expected)


# ---------------------------------------------------------------------------
# Shared-weight (parameter reuse across two independent sub-graphs)
#
#  W used in two matmul branches, losses summed.
#  dL/dW must accumulate both contributions.
# ---------------------------------------------------------------------------


class TestSharedWeights:
    def test_weight_reuse_gradient_accumulated(self):
        W = _T(np.ones((3, 3)) * 0.5, name="W")
        x1 = _T(np.eye(3), name="x1")
        x2 = _T(np.ones((3, 3)) * 2.0, name="x2")

        out1 = ops.matmul(x1, W)  # x1 @ W
        out2 = ops.matmul(x2, W)  # x2 @ W
        loss = ops.sum(ops.add(out1, out2))
        loss.backward()

        # d(sum(x1@W))/dW = x1.T  (sum over all outputs)
        # d(sum(x2@W))/dW = x2.T
        expected_dW = x1.data.T @ np.ones((3, 3)) + x2.data.T @ np.ones((3, 3))
        assert close(W.grad, expected_dW)

    def test_shared_bias_gradient(self):
        """bias b added to two separate activations."""
        b = _T(np.zeros(4), name="b")
        x = _T(np.random.randn(4), name="x")

        out1 = ops.add(x, b)  # x + b
        out2 = ops.multiply(x, ops.add(x, b))  # x * (x + b)
        loss = ops.sum(ops.add(out1, out2))
        loss.backward()

        x_np = x.data.copy()
        # d(sum(x+b))/db = 1 each element
        # d(sum(x*(x+b)))/db = x each element
        expected_db = np.ones(4) + x_np
        assert close(b.grad, expected_db)


# ---------------------------------------------------------------------------
# Self-product  f(x) = x * x,  df/dx = 2x
# ---------------------------------------------------------------------------


class TestSelfProduct:
    def test_x_times_x_gradient(self):
        x = _T([2.0, -3.0, 0.5])
        y = ops.multiply(x, x)
        loss = ops.sum(y)
        loss.backward()
        assert close(x.grad, 2 * x.data)

    def test_x_squared_via_power(self):
        x = _T([1.0, 2.0, 3.0])
        y = ops.power(x, _T([2.0, 2.0, 2.0]))
        loss = ops.sum(y)
        loss.backward()
        # gradient of x^2 w.r.t. x = 2x
        assert close(x.grad, 2 * x.data)


# ---------------------------------------------------------------------------
# Multi-use leaf: x used in 3 branches
# f = sum(x) + sum(x^2) + sum(exp(x))
# df/dx = 1 + 2x + exp(x)
# ---------------------------------------------------------------------------


class TestMultiUseLead:
    def test_three_branch_gradient(self):
        x_np = np.array([0.5, 1.0, -1.0])
        x = _T(x_np)

        l1 = ops.sum(x)
        l2 = ops.sum(ops.square(x))
        l3 = ops.sum(ops.exp(x))
        loss = ops.add(ops.add(l1, l2), l3)
        loss.backward()

        expected = np.ones(3) + 2 * x_np + np.exp(x_np)
        assert close(x.grad, expected)


# ---------------------------------------------------------------------------
# Quadratic: f(x) = (ax + b)^2, verified against finite differences
# ---------------------------------------------------------------------------


class TestFiniteDifference:
    @pytest.mark.parametrize(
        "x_np",
        [
            np.array([1.0, -2.0, 0.5, 3.0]),
            np.array([0.0]),
            np.random.default_rng(42).standard_normal(8),
        ],
    )
    def test_quadratic_finite_diff(self, x_np):
        a_np = np.random.default_rng(0).standard_normal(x_np.shape)
        b_np = np.random.default_rng(1).standard_normal(x_np.shape)

        def f_np(xv):
            return np.sum((a_np * xv + b_np) ** 2)

        x = _T(x_np.copy())
        a = _T(a_np)
        b = _T(b_np)
        inner = ops.add(ops.multiply(a, x), b)
        loss = ops.sum(ops.square(inner))
        loss.backward()

        # Analytical: d/dx[(ax+b)^2] = 2a(ax+b)
        analytical_x = x.grad.copy()
        numerical_x = np.array([_finite_diff(f_np, x_np, i) for i in range(x_np.size)])
        assert close(analytical_x, numerical_x), (
            f"x grad mismatch:\n  analytical={analytical_x}\n  numerical ={numerical_x}"
        )

    def test_quadratic_grad_a(self):
        """Gradient w.r.t. 'a' in (ax+b)^2."""
        x_np = np.array([2.0, -1.0, 3.0])
        a_np = np.array([1.0, 2.0, -1.0])
        b_np = np.array([0.5, -0.5, 1.0])

        def f_np(av):
            return np.sum((av * x_np + b_np) ** 2)

        x = _T(x_np)
        a = _T(a_np.copy())
        b = _T(b_np)
        inner = ops.add(ops.multiply(a, x), b)
        loss = ops.sum(ops.square(inner))
        loss.backward()

        numerical_a = np.array([_finite_diff(f_np, a_np, i) for i in range(a_np.size)])
        assert close(a.grad, numerical_a)


# ---------------------------------------------------------------------------
# Three-branch fan-out then recombine
#
#   x → branch1 = relu(x)
#   x → branch2 = tanh(x)
#   x → branch3 = sigmoid(x)
#   loss = sum(b1 + b2 + b3)
#
# df/dx = (x>0) + (1-tanh²(x)) + sigmoid(x)*(1-sigmoid(x))
# ---------------------------------------------------------------------------


class TestThreeBranchFanOut:
    def test_three_forward_activations(self):
        x_np = np.array([-1.5, 0.0, 1.5, 3.0])
        x = _T(x_np.copy())

        b1 = ops.relu(x)
        b2 = ops.tanh(x)
        b3 = ops.sigmoid(x)
        loss = ops.sum(ops.add(ops.add(b1, b2), b3))
        loss.backward()

        sig = 1.0 / (1.0 + np.exp(-x_np))
        expected = (
            (x_np > 0).astype(float) + (1.0 - np.tanh(x_np) ** 2) + sig * (1.0 - sig)
        )
        assert close(x.grad, expected)


# ---------------------------------------------------------------------------
# Scalar broadcast: s added to every element of a matrix
# ---------------------------------------------------------------------------


class TestScalarBroadcast:
    def test_scalar_broadcast_add(self):
        """Adding a scalar Tensor to a matrix: d(loss)/d(scalar) = n_elements."""
        s = _T(np.array(2.0), name="s")
        M = _T(np.ones((3, 4)), name="M")
        out = ops.add(M, s)
        loss = ops.sum(out)
        loss.backward()

        # d(sum(M + s))/ds = 12 (all 3*4 elements)
        assert close(s.grad, np.array(12.0))
        assert close(M.grad, np.ones((3, 4)))

    def test_scalar_broadcast_multiply(self):
        """Multiplying a scalar by a matrix — gradient is sum over broadcasts."""
        s = _T(np.array(3.0), name="s")
        M = _T(np.ones((2, 5)), name="M")
        out = ops.multiply(s, M)
        loss = ops.sum(out)
        loss.backward()

        # d(sum(s*M))/ds = sum(M)
        assert close(s.grad, np.sum(M.data))
        # d(sum(s*M))/dM = s (broadcast)
        assert close(M.grad, np.full((2, 5), 3.0))


# ---------------------------------------------------------------------------
# Deep chain: loss = f(f(f(f(f(x))))) where f(x) = relu(x + 0.1)
# Gradient should never vanish (relu has grad=1 for positive inputs)
# ---------------------------------------------------------------------------


class TestDeepChain:
    def test_depth_20_gradient_flows(self):
        x = _T(np.array([1.0, 2.0, 3.0]))
        out = x
        depth = 20
        for _ in range(depth):
            out = ops.relu(ops.add(out, _T(np.array([0.1, 0.1, 0.1]))))
        loss = ops.sum(out)
        loss.backward()

        # ReLU is active throughout (all positive), so df/dx = 1 every step
        assert close(x.grad, np.ones(3)), f"deep chain grad: {x.grad}"

    def test_gradient_accumulation_correct_in_long_chain(self):
        """f(x) = x + x + x + ... (n times).  df/dx = n."""
        x = _T(np.array([1.0]))
        out = x
        n = 15
        for _ in range(n):
            out = ops.add(out, x)  # NOTE: x is re-used each step
        loss = ops.sum(out)
        loss.backward()

        # Each add contributes +1 to x.grad, plus the final passthrough
        # out = x + x + ... (n adds starting from x) = (n+1)*x
        assert close(x.grad, np.array([float(n + 1)]))


# ---------------------------------------------------------------------------
# MLP end-to-end: 2-layer network, cross-entropy loss, check grads via FD
# ---------------------------------------------------------------------------


class TestMLPEndToEnd:
    def test_mlp_backward_matches_finite_diff(self):
        """Gradient of a 2-layer MLP w.r.t. the first-layer weights."""
        np.random.seed(99)

        W1_np = np.random.randn(4, 8) * 0.5
        b1_np = np.zeros(8)
        W2_np = np.random.randn(8, 3) * 0.5
        b2_np = np.zeros(3)
        x_np = np.random.randn(2, 4)  # batch=2
        # one-hot targets
        y_np = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

        def _forward(W1v):
            W1 = _T(W1v, name="W1")
            b1 = _T(b1_np, name="b1")
            W2 = _T(W2_np, name="W2")
            b2 = _T(b2_np, name="b2")
            x = _T(x_np, name="x")
            y = _T(y_np, name="y")

            h = ops.relu(ops.add(ops.matmul(x, W1), b1))
            logits = ops.add(ops.matmul(h, W2), b2)
            probs = ops.softmax(logits)
            loss = ops.mean(
                ops.negative(
                    ops.sum(
                        ops.multiply(
                            y, ops.log(ops.add(probs, _T(np.full((2, 3), 1e-8))))
                        ),
                        axis=-1,
                    )
                )
            )
            return float(loss.data)

        def _scalar_loss(W1v):
            return _forward(W1v)

        # Autograd gradient for W1
        W1 = _T(W1_np.copy(), name="W1")
        b1 = _T(b1_np, name="b1")
        W2 = _T(W2_np, name="W2")
        b2 = _T(b2_np, name="b2")
        x = _T(x_np, name="x")
        y = _T(y_np, name="y")

        h = ops.relu(ops.add(ops.matmul(x, W1), b1))
        logits = ops.add(ops.matmul(h, W2), b2)
        probs = ops.softmax(logits)
        eps_t = _T(np.full((2, 3), 1e-8))
        loss = ops.mean(
            ops.negative(
                ops.sum(ops.multiply(y, ops.log(ops.add(probs, eps_t))), axis=-1)
            )
        )
        loss.backward()

        auto_grad = W1.grad.copy()

        # Numerical gradient for a few entries of W1
        for idx in np.ndindex(W1_np.shape):
            num = _finite_diff(
                _scalar_loss, W1_np, np.ravel_multi_index(idx, W1_np.shape)
            )
            assert abs(auto_grad[idx] - num) < 1e-4, (
                f"W1 grad mismatch at {idx}: autograd={auto_grad[idx]:.6f}, FD={num:.6f}"
            )
