"""
dnp/core/ops.py
===============
`Ops` class and all pre-built operation instances.

Each `Ops` object bundles:
  - a forward callable  (`op.func`)
  - its VJP rule       (`op.vpj`)
  - a human-readable name

Previously split between dnp/ops/numpy_ops.py and dnp/ops/vjp_rules.py.
"""

# Third-party libraries
import numpy as np

# Local imports
from .vjp_rules import (
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
    conv2d,
    rot180,
    conv2d_full,
    max_pool2d,
    avg_pool2d,
    dropout,
    batch_norm,
    _reshape,
)


class Ops:
    """Bundles a forward function with its VJP rule under a single object."""

    def __init__(self, func, vpj_fun, name=None):
        self.func = func
        self.vpj_fun = vpj_fun
        self.name = name or getattr(func, "__name__", repr(func))

    def __call__(self, *args, **kwargs):
        from .tensor import Tensor

        # Extract raw data
        args_data = [a.data if isinstance(a, Tensor) else a for a in args]
        kwargs_data = {
            k: (v.data if isinstance(v, Tensor) else v) for k, v in kwargs.items()
        }

        out_data = self.func(*args_data, **kwargs_data)

        parents = [a for a in args if isinstance(a, Tensor)]
        parents.extend([v for v in kwargs.values() if isinstance(v, Tensor)])

        op_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, Tensor)}

        if not parents:
            return out_data

        return Tensor(
            out_data,
            parents=parents,
            op_func=self.func,
            op_kwargs=op_kwargs,
            name=self.name,
        )

    def vpj(self, *args, **kwargs):
        return self.vpj_fun(*args, **kwargs)

    def __repr__(self):
        return f"Ops('{self.name}')"


# ---------------------------------------------------------------------------
# Binary operations
# ---------------------------------------------------------------------------
add = Ops(np.add, VJP_RULES[np.add], name="add")
subtract = Ops(np.subtract, VJP_RULES[np.subtract], name="subtract")
multiply = Ops(np.multiply, VJP_RULES[np.multiply], name="multiply")
divide = Ops(np.divide, VJP_RULES[np.divide], name="divide")
power = Ops(np.power, VJP_RULES[np.power], name="power")
maximum = Ops(np.maximum, VJP_RULES[np.maximum], name="maximum")
minimum = Ops(np.minimum, VJP_RULES[np.minimum], name="minimum")
matmul = Ops(np.matmul, VJP_RULES[np.matmul], name="matmul")
dot = Ops(np.dot, VJP_RULES[np.dot], name="dot")

# ---------------------------------------------------------------------------
# Unary operations
# ---------------------------------------------------------------------------
negative = Ops(np.negative, VJP_RULES[np.negative], name="negative")
square = Ops(np.square, VJP_RULES[np.square], name="square")
sqrt = Ops(np.sqrt, VJP_RULES[np.sqrt], name="sqrt")
exp = Ops(np.exp, VJP_RULES[np.exp], name="exp")
log = Ops(np.log, VJP_RULES[np.log], name="log")
log1p = Ops(np.log1p, VJP_RULES[np.log1p], name="log1p")
expm1 = Ops(np.expm1, VJP_RULES[np.expm1], name="expm1")
absolute = Ops(np.abs, VJP_RULES[np.abs], name="abs")
sign = Ops(np.sign, VJP_RULES[np.sign], name="sign")

# Trigonometry & Hyperbolic
sin = Ops(np.sin, VJP_RULES[np.sin], name="sin")
cos = Ops(np.cos, VJP_RULES[np.cos], name="cos")
tan = Ops(np.tan, VJP_RULES[np.tan], name="tan")
sinh = Ops(np.sinh, VJP_RULES[np.sinh], name="sinh")
cosh = Ops(np.cosh, VJP_RULES[np.cosh], name="cosh")
tanh = Ops(np.tanh, VJP_RULES[np.tanh], name="tanh")

# Rounding
floor = Ops(np.floor, VJP_RULES[np.floor], name="floor")
ceil = Ops(np.ceil, VJP_RULES[np.ceil], name="ceil")
round = Ops(np.round, VJP_RULES[np.round], name="round")

# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------
sum = Ops(np.sum, VJP_RULES[np.sum], name="sum")
mean = Ops(np.mean, VJP_RULES[np.mean], name="mean")
prod = Ops(np.prod, VJP_RULES[np.prod], name="prod")
max = Ops(np.max, VJP_RULES[np.max], name="max")
min = Ops(np.min, VJP_RULES[np.min], name="min")

# ---------------------------------------------------------------------------
# Shape operations
# ---------------------------------------------------------------------------
reshape = Ops(_reshape, VJP_RULES[_reshape], name="reshape")
transpose = Ops(np.transpose, VJP_RULES[np.transpose], name="transpose")
expand_dims = Ops(np.expand_dims, VJP_RULES[np.expand_dims], name="expand_dims")
squeeze = Ops(np.squeeze, VJP_RULES[np.squeeze], name="squeeze")

# ---------------------------------------------------------------------------
# Neural-network & custom ops
# ---------------------------------------------------------------------------
conv2d = Ops(conv2d, VJP_RULES[conv2d], name="conv2d")
sigmoid = Ops(sigmoid, VJP_RULES[sigmoid], name="sigmoid")
relu = Ops(relu, VJP_RULES[relu], name="relu")
leaky_relu = Ops(leaky_relu, VJP_RULES[leaky_relu], name="leaky_relu")
elu = Ops(elu, VJP_RULES[elu], name="elu")
softplus = Ops(softplus, VJP_RULES[softplus], name="softplus")
swish = Ops(swish, VJP_RULES[swish], name="swish")
gelu = Ops(gelu, VJP_RULES[gelu], name="gelu")
softmax = Ops(softmax, VJP_RULES[softmax], name="softmax")

# Pooling operations (vectorized)
max_pool2d = Ops(max_pool2d, VJP_RULES[max_pool2d], name="max_pool2d")
avg_pool2d = Ops(avg_pool2d, VJP_RULES[avg_pool2d], name="avg_pool2d")

# Regularization operations
dropout = Ops(dropout, VJP_RULES[dropout], name="dropout")

# Normalization operations
batch_norm = Ops(batch_norm, VJP_RULES[batch_norm], name="batch_norm")


__all__ = [
    "Ops",
    # binary
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "maximum",
    "minimum",
    "matmul",
    "dot",
    # unary
    "negative",
    "square",
    "sqrt",
    "exp",
    "log",
    "log1p",
    "expm1",
    "absolute",
    "sign",
    # trig
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    # rounding
    "floor",
    "ceil",
    "round",
    # reductions
    "sum",
    "mean",
    "prod",
    "max",
    "min",
    # shape
    "reshape",
    "transpose",
    "expand_dims",
    "squeeze",
    # nn
    "conv2d",
    "sigmoid",
    "relu",
    "leaky_relu",
    "elu",
    "softplus",
    "swish",
    "gelu",
    "softmax",
    # pooling (vectorized)
    "max_pool2d",
    "avg_pool2d",
    # regularization
    "dropout",
    # normalization
    "batch_norm",
]
