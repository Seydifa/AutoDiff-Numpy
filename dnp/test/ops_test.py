"""
ops_test.py
===========
Tests for dnp/core/ops.py:
  - Ops class behaviour (forward call, vpj method, repr)
  - Forward pass of every Ops instance
  - vpj() delegation matches VJP_RULES
  - All Ops instances exported from dnp.core
"""

# Third-party
import pytest
import numpy as np

# Local imports
from dnp.core.ops import (
    Ops,
    add,
    subtract,
    multiply,
    divide,
    power,
    maximum,
    minimum,
    matmul,
    dot,
    negative,
    square,
    sqrt,
    exp,
    log,
    log1p,
    expm1,
    absolute,
    sign,
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    floor,
    ceil,
    round,
    sum,
    mean,
    prod,
    max,
    min,
    reshape,
    transpose,
    expand_dims,
    squeeze,
    conv2d,
    sigmoid,
    relu,
    leaky_relu,
    elu,
    softplus,
    swish,
    gelu,
    softmax,
)
from dnp.core.vjp_rules import VJP_RULES

close = lambda a, b: np.allclose(a, b, atol=1e-5)


# ===========================================================================
# Ops class
# ===========================================================================


class TestOpsClass:
    def test_forward_call(self):
        op = Ops(np.add, name="test_add")
        result = op(np.array([1.0]), np.array([2.0]))
        assert close(result, np.array([3.0]))

    def test_vpj_delegation(self):
        op = Ops(np.add, name="test_add")
        gx, gy = op.vpj(np.ones(2), np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert close(gx, np.ones(2)) and close(gy, np.ones(2))

    def test_name_attribute(self):
        op = Ops(np.add, name="my_op")
        assert op.name == "my_op"

    def test_repr_contains_name(self):
        op = Ops(np.exp, name="exp")
        assert "exp" in repr(op)

    def test_auto_name_from_func(self):
        op = Ops(np.sqrt)
        assert op.name == "sqrt"


# ===========================================================================
# Forward pass of all Ops instances
# ===========================================================================


class TestForwardPass:
    x = np.array([1.0, 4.0])
    y = np.array([2.0, 2.0])

    def test_add(self):
        assert close(add(self.x, self.y), np.array([3.0, 6.0]))

    def test_subtract(self):
        assert close(subtract(self.x, self.y), np.array([-1.0, 2.0]))

    def test_multiply(self):
        assert close(multiply(self.x, self.y), np.array([2.0, 8.0]))

    def test_divide(self):
        assert close(divide(self.x, self.y), np.array([0.5, 2.0]))

    def test_power(self):
        assert close(power(self.x, self.y), np.array([1.0, 16.0]))

    def test_maximum(self):
        assert close(maximum(self.x, self.y), np.array([2.0, 4.0]))

    def test_minimum(self):
        assert close(minimum(self.x, self.y), np.array([1.0, 2.0]))

    def test_negative(self):
        assert close(negative(self.x), -self.x)

    def test_square(self):
        assert close(square(self.x), np.array([1.0, 16.0]))

    def test_sqrt(self):
        assert close(sqrt(self.x), np.array([1.0, 2.0]))

    def test_exp(self):
        assert close(exp(np.array([0.0])), np.array([1.0]))

    def test_log(self):
        assert close(log(np.array([1.0])), np.array([0.0]))

    def test_log1p(self):
        assert close(log1p(np.array([0.0])), np.array([0.0]))

    def test_expm1(self):
        assert close(expm1(np.array([0.0])), np.array([0.0]))

    def test_absolute(self):
        assert close(absolute(np.array([-3.0])), np.array([3.0]))

    def test_sign(self):
        assert close(sign(np.array([-2.0, 0.0, 3.0])), np.array([-1.0, 0.0, 1.0]))

    def test_sin(self):
        assert close(sin(np.array([0.0])), np.array([0.0]))

    def test_cos(self):
        assert close(cos(np.array([0.0])), np.array([1.0]))

    def test_tanh(self):
        assert close(tanh(np.array([0.0])), np.array([0.0]))

    def test_floor(self):
        assert close(floor(np.array([1.7])), np.array([1.0]))

    def test_ceil(self):
        assert close(ceil(np.array([1.2])), np.array([2.0]))

    def test_round(self):
        assert close(round(np.array([1.5])), np.array([2.0]))

    def test_sum(self):
        assert close(sum(self.x), np.array(5.0))

    def test_mean(self):
        assert close(mean(self.x), np.array(2.5))

    def test_prod(self):
        assert close(prod(self.x), np.array(4.0))

    def test_max(self):
        assert close(max(self.x), np.array(4.0))

    def test_min(self):
        assert close(min(self.x), np.array(1.0))

    def test_sigmoid(self):
        assert close(sigmoid(np.array([0.0])), np.array([0.5]))

    def test_relu(self):
        assert close(relu(np.array([-1.0, 2.0])), np.array([0.0, 2.0]))

    def test_softmax(self):
        x = np.array([[1.0, 2.0, 3.0]])
        assert close(softmax(x).data.sum(), 1.0)

    def test_matmul_shape(self):
        A = np.ones((2, 3))
        B = np.ones((3, 4))
        assert matmul(A, B).shape == (2, 4)

    def test_dot_scalar(self):
        assert close(dot(np.array([1.0, 2.0]), np.array([3.0, 4.0])), np.array(11.0))

    def test_reshape_shape(self):
        x = np.ones((2, 3))
        assert reshape(x, (6,)).shape == (6,)

    def test_transpose_shape(self):
        x = np.ones((3, 4))
        assert transpose(x).shape == (4, 3)

    def test_expand_dims_shape(self):
        x = np.ones((3, 4))
        assert expand_dims(x, 0).shape == (1, 3, 4)

    def test_squeeze_shape(self):
        x = np.ones((1, 3, 1))
        assert squeeze(x).shape == (3,)

    def test_conv2d_shape(self):
        x = np.random.randn(5, 5)
        w = np.random.randn(3, 3)
        assert conv2d(x, w, mode="valid").shape == (3, 3)


# ===========================================================================
# VJP delegation — Ops.vpj delegates correctly to VJP_RULES
# ===========================================================================


class TestVPJDelegation:
    """Ensuring op.vpj() returns identical results to VJP_RULES lookup."""

    g = np.ones(3)
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    def _check(self, op, np_key, *args):
        via_op = op.vpj(*args)
        via_rules = VJP_RULES[np_key](*args)
        for a, b in zip(via_op, via_rules):
            assert close(a, b)

    def test_add_vpj(self):
        self._check(add, np.add, self.g, self.x, self.y)

    def test_subtract_vpj(self):
        self._check(subtract, np.subtract, self.g, self.x, self.y)

    def test_multiply_vpj(self):
        self._check(multiply, np.multiply, self.g, self.x, self.y)

    def test_exp_vpj(self):
        self._check(exp, np.exp, self.g, self.x)

    def test_log_vpj(self):
        self._check(log, np.log, self.g, self.x)

    def test_sum_vpj(self):
        self._check(sum, np.sum, np.array(1.0), self.x)


# ===========================================================================
# All Ops instances are importable from dnp.core
# ===========================================================================


class TestCoreExports:
    def test_all_ops_importable_from_core(self):
        import dnp.core as core

        ops_list = [
            "add",
            "subtract",
            "multiply",
            "divide",
            "power",
            "maximum",
            "minimum",
            "matmul",
            "dot",
            "negative",
            "square",
            "sqrt",
            "exp",
            "log",
            "log1p",
            "expm1",
            "absolute",
            "sign",
            "sin",
            "cos",
            "tan",
            "sinh",
            "cosh",
            "tanh",
            "floor",
            "ceil",
            "round",
            "sum",
            "mean",
            "prod",
            "max",
            "min",
            "reshape",
            "transpose",
            "expand_dims",
            "squeeze",
            "conv2d",
            "sigmoid",
            "relu",
            "leaky_relu",
            "elu",
            "softplus",
            "swish",
            "gelu",
            "softmax",
        ]
        for name in ops_list:
            assert hasattr(core, name), f"dnp.core missing: {name}"
            assert isinstance(getattr(core, name), Ops), (
                f"{name} is not an Ops instance"
            )
