"""
dnp/core/ops.py
===============
`Ops` class and all pre-built operation instances.

Each `Ops` object bundles:
  - a forward callable  (`op.func`)
  - a human-readable name

v3 changes
----------
* Every ``Ops.__call__`` is now wrapped by :func:`graph_fn`, so raw numpy/cupy
  arrays and plain Python scalars are auto-promoted to ``Tensor`` before the
  forward function runs.  ``array_creation`` helpers (arange, ones, zeros, …)
  are also exposed here as graph-compatible callables.
* ``vpj_fun`` attribute removed — it was stored but never used; VJPs are
  looked up directly via ``VJP_RULES[node.op_func]`` in ``Tensor.backward()``.
"""

# Third-party libraries
import numpy as np

# Local imports
from .session import graph_fn
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
    """Bundles a forward function into a graph-compatible callable.

    v3: every call automatically promotes raw arrays / scalars to ``Tensor``
    via the ``graph_fn`` decorator, so users can write:

        import numpy as np
        x = np.array([1.0, 2.0, 3.0])
        y = ops.sin(x)   # x is converted to Tensor, y is a Tensor
    """

    def __init__(self, func, name=None):
        self.func = func
        self.name = name or getattr(func, "__name__", repr(func))
        # Build the graph-fn-wrapped inner call once at construction time.
        self._graph_call = graph_fn(self._raw_call)

    def _raw_call(self, *args, **kwargs):
        """Inner call that expects Tensor arguments (graph_fn handles promotion)."""
        from .tensor import Tensor

        # Extract raw data
        args_data = [a.data if isinstance(a, Tensor) else a for a in args]
        kwargs_data = {
            k: (v.data if isinstance(v, Tensor) else v) for k, v in kwargs.items()
        }

        out_data = self.func(*args_data, **kwargs_data)

        # Collect Tensor parents (positional only; kwargs Tensors are rare and
        # handled separately via op_kwargs).
        parent_indices = [i for i, a in enumerate(args) if isinstance(a, Tensor)]
        parents = [args[i] for i in parent_indices]
        parents.extend([v for v in kwargs.values() if isinstance(v, Tensor)])

        op_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, Tensor)}

        # When there are non-Tensor positional args mixed with Tensor args (e.g.
        # subtract(1.0, tensor)), the backward pass needs the full args list to
        # call the VJP rule correctly and to map gradients to the right parents.
        if len(parent_indices) < len(args):
            op_kwargs["_vjp_args"] = args_data
            op_kwargs["_vjp_parent_indices"] = parent_indices

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
        """Look up and call the VJP rule for this op.

        Signature mirrors VJP_RULES: ``vpj(grad, *inputs, **op_kwargs)``.
        This is provided for backward compatibility and introspection.
        """
        from .vjp_rules import VJP_RULES

        rule = VJP_RULES.get(self.func)
        if rule is None:
            raise KeyError(f"No VJP rule found for '{self.name}'")
        return rule(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._graph_call(*args, **kwargs)

    def __repr__(self):
        return f"Ops('{self.name}')"


# ---------------------------------------------------------------------------
# Specialised Dropout Op — saves binary mask in op_kwargs for VJP
# ---------------------------------------------------------------------------


class _DropoutOps(Ops):
    """Specialised Ops for dropout that saves the binary mask in op_kwargs.

    During the forward pass the mask is generated here and stored in
    ``op_kwargs["mask"]`` so the VJP (``VJP_RULES[dropout_func]``) can
    reuse exactly the same mask instead of dividing uniformly by ``(1-p)``.
    """

    def _raw_call(self, *args, **kwargs):
        from .tensor import Tensor
        from .backend import get_xp

        x = args[0]
        p = kwargs.get("p", 0.5)
        training = kwargs.get("training", True)

        x_data = x.data if isinstance(x, Tensor) else x

        if training and p > 0:
            xp = get_xp(x_data)
            mask = (xp.random.uniform(size=x_data.shape) >= p).astype(x_data.dtype)
            out_data = x_data * mask / (1.0 - p)
        else:
            mask = None
            out_data = x_data

        parents = [x] if isinstance(x, Tensor) else []
        if not parents:
            return out_data

        return Tensor(
            out_data,
            parents=parents,
            op_func=self.func,
            op_kwargs={"p": p, "training": training, "mask": mask},
            name=self.name,
        )


# ---------------------------------------------------------------------------
# Binary operations
# ---------------------------------------------------------------------------
add = Ops(np.add, name="add")
subtract = Ops(np.subtract, name="subtract")
multiply = Ops(np.multiply, name="multiply")
divide = Ops(np.divide, name="divide")
power = Ops(np.power, name="power")
maximum = Ops(np.maximum, name="maximum")
minimum = Ops(np.minimum, name="minimum")
matmul = Ops(np.matmul, name="matmul")
dot = Ops(np.dot, name="dot")

# ---------------------------------------------------------------------------
# Unary operations
# ---------------------------------------------------------------------------
negative = Ops(np.negative, name="negative")
square = Ops(np.square, name="square")
sqrt = Ops(np.sqrt, name="sqrt")
exp = Ops(np.exp, name="exp")
log = Ops(np.log, name="log")
log1p = Ops(np.log1p, name="log1p")
expm1 = Ops(np.expm1, name="expm1")
absolute = Ops(np.abs, name="abs")
sign = Ops(np.sign, name="sign")

# Trigonometry & Hyperbolic
sin = Ops(np.sin, name="sin")
cos = Ops(np.cos, name="cos")
tan = Ops(np.tan, name="tan")
sinh = Ops(np.sinh, name="sinh")
cosh = Ops(np.cosh, name="cosh")
tanh = Ops(np.tanh, name="tanh")

# Rounding
floor = Ops(np.floor, name="floor")
ceil = Ops(np.ceil, name="ceil")
round = Ops(np.round, name="round")

# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------
sum = Ops(np.sum, name="sum")
mean = Ops(np.mean, name="mean")
prod = Ops(np.prod, name="prod")
max = Ops(np.max, name="max")
min = Ops(np.min, name="min")

# ---------------------------------------------------------------------------
# Shape operations
# ---------------------------------------------------------------------------
reshape = Ops(_reshape, name="reshape")
transpose = Ops(np.transpose, name="transpose")
expand_dims = Ops(np.expand_dims, name="expand_dims")
squeeze = Ops(np.squeeze, name="squeeze")

# ---------------------------------------------------------------------------
# Neural-network & custom ops
# ---------------------------------------------------------------------------
conv2d = Ops(conv2d, name="conv2d")
sigmoid = Ops(sigmoid, name="sigmoid")
relu = Ops(relu, name="relu")
leaky_relu = Ops(leaky_relu, name="leaky_relu")
elu = Ops(elu, name="elu")
softplus = Ops(softplus, name="softplus")
swish = Ops(swish, name="swish")
gelu = Ops(gelu, name="gelu")
softmax = Ops(softmax, name="softmax")

# Pooling operations (vectorized)
max_pool2d = Ops(max_pool2d, name="max_pool2d")
avg_pool2d = Ops(avg_pool2d, name="avg_pool2d")

# Regularization operations — use _DropoutOps to capture the mask
dropout = _DropoutOps(dropout, name="dropout")

# Normalization operations
batch_norm = Ops(batch_norm, name="batch_norm")


# ---------------------------------------------------------------------------
# Array creation helpers — return Tensors so they are graph-compatible
# ---------------------------------------------------------------------------


def _tensor_creation(np_func, name):
    """Factory: wrap a numpy array-creation function to return a leaf Tensor."""
    from .tensor import Tensor

    @graph_fn.__wrapped__ if hasattr(graph_fn, "__wrapped__") else (lambda f: f)
    def _creator(*args, **kwargs):
        data = np_func(*args, **kwargs)
        return Tensor(data, name=name)

    # Build a plain wrapper (no graph_fn autocast needed — these create leaves)
    def creator(*args, **kwargs):
        from .tensor import Tensor

        data = np_func(*args, **kwargs)
        return Tensor(data, name=name)

    creator.__name__ = name
    creator.__qualname__ = name
    return creator


arange = _tensor_creation(np.arange, "arange")
linspace = _tensor_creation(np.linspace, "linspace")
ones = _tensor_creation(np.ones, "ones")
zeros = _tensor_creation(np.zeros, "zeros")
full = _tensor_creation(np.full, "full")
eye = _tensor_creation(np.eye, "eye")


def ones_like(x):
    """Return a Tensor of ones with the same shape/device as *x*."""
    from .tensor import Tensor
    from .backend import get_xp

    xp = get_xp(x.data if isinstance(x, Tensor) else x)
    src = x.data if isinstance(x, Tensor) else x
    return Tensor(xp.ones_like(src), name="ones_like")


def zeros_like(x):
    """Return a Tensor of zeros with the same shape/device as *x*."""
    from .tensor import Tensor
    from .backend import get_xp

    xp = get_xp(x.data if isinstance(x, Tensor) else x)
    src = x.data if isinstance(x, Tensor) else x
    return Tensor(xp.zeros_like(src), name="zeros_like")


def full_like(x, fill_value):
    """Return a Tensor filled with *fill_value*, same shape/device as *x*."""
    from .tensor import Tensor
    from .backend import get_xp

    xp = get_xp(x.data if isinstance(x, Tensor) else x)
    src = x.data if isinstance(x, Tensor) else x
    return Tensor(xp.full_like(src, fill_value), name="full_like")


# ---------------------------------------------------------------------------
# Random module — returns Tensor
# ---------------------------------------------------------------------------


class _RandomModule:
    """Namespace for random array creation that returns Tensors."""

    @staticmethod
    def randn(*shape):
        from .tensor import Tensor

        return Tensor(np.random.randn(*shape), name="randn")

    @staticmethod
    def rand(*shape):
        from .tensor import Tensor

        return Tensor(np.random.rand(*shape), name="rand")

    @staticmethod
    def randint(low, high=None, size=None):
        from .tensor import Tensor

        return Tensor(
            np.random.randint(low, high=high, size=size).astype(np.float64),
            name="randint",
        )

    @staticmethod
    def uniform(low=0.0, high=1.0, size=None):
        from .tensor import Tensor

        return Tensor(np.random.uniform(low, high, size), name="uniform")

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None):
        from .tensor import Tensor

        return Tensor(np.random.normal(loc, scale, size), name="normal")

    @staticmethod
    def seed(s):
        np.random.seed(s)


random = _RandomModule()


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
    # array creation
    "arange",
    "linspace",
    "ones",
    "zeros",
    "full",
    "eye",
    "ones_like",
    "zeros_like",
    "full_like",
    # random
    "random",
]
