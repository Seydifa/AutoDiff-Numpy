"""
dnp/core/ops.py
===============
``Ops`` class and all pre-built operation instances.

Each ``Ops`` object bundles:
  - a forward callable (``op.func``) — the NumPy ufunc / function used as the
    ``VJP_RULES`` registry key, guaranteeing unique identity across backends.
  - explicit backend dispatch at call time (CuPy when arrays live on GPU)
  - a ``graph_fn`` wrapper that auto-promotes raw arrays / scalars to ``Tensor``

Sections
--------
1. ``Ops`` base class + specialised subclasses
   (``_DropoutOps``, ``_GatherOps``, ``_WhereOps``)
2. Pre-built op instances grouped by category
   binary · unary · trig · rounding · reductions · shape ·
   array-manipulation · nn activations · conv · pooling · norm
3. Array creation helpers (``arange``, ``ones``, ``zeros``, …)
4. ``_RandomModule``  (``ops.random.randn``, …)
5. ``__all__``
"""

# Third-party libraries
import numpy as np

# Local imports
from .session import graph_fn
from .vjp_rules import (
    VJP_RULES,
    # forward callables used as func= for Ops instances below
    sigmoid,
    relu,
    leaky_relu,
    elu,
    softplus,
    swish,
    gelu,
    softmax,
    conv2d,
    max_pool2d,
    avg_pool2d,
    dropout,
    batch_norm,
    _reshape,
    _concatenate,
    _stack,
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
        from .backend import get_xp

        # Extract raw data
        args_data = [a.data if isinstance(a, Tensor) else a for a in args]
        kwargs_data = {
            k: (v.data if isinstance(v, Tensor) else v) for k, v in kwargs.items()
        }

        # Explicitly dispatch to the correct backend module.
        # self.func stays as np.* so VJP_RULES lookup keys are never invalidated.
        # Fallback to self.func (protocol dispatch) when no named match is found.
        first_arr = next(
            (a for a in args_data if hasattr(a, "shape")),
            next((v for v in kwargs_data.values() if hasattr(v, "shape")), None),
        )
        if first_arr is not None:
            xp = get_xp(first_arr)
            if xp is not np:
                func_name = getattr(self.func, "__name__", None)
                xp_func = getattr(xp, func_name, None) if func_name else None
                _call = xp_func if xp_func is not None else self.func
            else:
                _call = self.func
        else:
            _call = self.func

        out_data = _call(*args_data, **kwargs_data)

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
            op_func=self,  # store the Ops instance — backward dispatches via self.vpj()
            op_kwargs=op_kwargs,
            name=self.name,
        )

    def vpj(self, grad, *args, **op_kwargs):
        """Compute and return the VJP tuple for this op.

        This is the authoritative VJP dispatch point for every ``Ops``
        instance.  ``Tensor.backward()`` calls ``op.vpj(...)`` directly
        so the rule lookup always goes through here — never bypassing the
        ``Ops`` object by hitting ``VJP_RULES`` directly.

        Handles the ``_vjp_args`` / ``_vjp_parent_indices`` keys that
        ``_raw_call`` inserts when positional args are a mix of scalars
        and Tensors (e.g. ``1.0 - tensor``).
        """
        from .backend import get_xp as _get_xp

        if "_vjp_args" in op_kwargs:
            # Reconstruct the full positional arg list; wrap plain scalars as
            # 0-d arrays on the same device as grad so VJP lambdas that call
            # .shape on all args work and get_xp() returns the right backend.
            _xp = _get_xp(grad)
            full_args = [
                _xp.asarray(a) if isinstance(a, (int, float)) else a
                for a in op_kwargs["_vjp_args"]
            ]
            clean_kwargs = {
                k: v
                for k, v in op_kwargs.items()
                if k not in ("_vjp_args", "_vjp_parent_indices")
            }
            rule = VJP_RULES.get(self.func)
            if rule is None:
                raise KeyError(f"No VJP rule found for '{self.name}'")
            return rule(grad, *full_args, **clean_kwargs)

        rule = VJP_RULES.get(self.func)
        if rule is None:
            raise KeyError(f"No VJP rule found for '{self.name}'")
        return rule(grad, *args, **op_kwargs)

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
# Gather Op — W[indices], differentiable w.r.t. W via scatter-add
# ---------------------------------------------------------------------------


def _gather_forward(W, indices):
    """Forward function for the gather op: ``output = W[indices]``."""
    return W[indices]


class _GatherOps(Ops):
    """Gather op — ``output = W[indices]``.

    Only *W* is differentiable; *indices* are integer arrays and carry no
    gradient.  The VJP is a scatter-add::

        dW[indices] += upstream_grad

    ``vpj()`` is overridden here so the backward is fully self-contained
    without needing a separate entry in the global ``VJP_RULES`` dict.
    This also means any ``Ops`` subclass can define its own backward by
    simply overriding ``vpj()``.
    """

    def _raw_call(self, *args, **kwargs):
        from .tensor import Tensor
        from .backend import as_numpy

        W = args[0]
        # indices may be a raw int array OR a Tensor (graph_fn converts numpy
        # arrays to Tensor automatically; we unwrap it back to plain ints).
        raw_idx = args[1] if len(args) > 1 else kwargs.get("indices")
        if isinstance(raw_idx, Tensor):
            indices = as_numpy(raw_idx.data).astype(int)
        else:
            indices = np.asarray(raw_idx, dtype=int)

        W_data = W.data if isinstance(W, Tensor) else W
        out_data = W_data[indices]

        parents = [W] if isinstance(W, Tensor) else []
        if not parents:
            return out_data

        return Tensor(
            out_data,
            parents=parents,
            op_func=self,  # backward dispatches via self.vpj()
            op_kwargs={"indices": indices},
            name=self.name,
        )

    def vpj(self, grad, W, **op_kwargs):
        """Scatter-add VJP: dW[indices] += grad."""
        from .backend import get_xp

        indices = op_kwargs["indices"]
        xp = get_xp(W)
        dW = xp.zeros_like(W)
        xp.add.at(dW, indices, grad)
        return (dW,)


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
# Array manipulation operations
# ---------------------------------------------------------------------------
concatenate = Ops(_concatenate, name="concatenate")
stack = Ops(_stack, name="stack")
clip = Ops(np.clip, name="clip")
cumsum = Ops(np.cumsum, name="cumsum")
flip = Ops(np.flip, name="flip")
roll = Ops(np.roll, name="roll")
tile = Ops(np.tile, name="tile")
repeat = Ops(np.repeat, name="repeat")


# ---------------------------------------------------------------------------
# where — condition is not differentiable; handled by a custom Ops subclass
# ---------------------------------------------------------------------------


class _WhereOps(Ops):
    """where(condition, x, y) — condition is excluded from gradient tracking.

    ``graph_fn`` would normally convert a boolean ndarray condition into a
    Tensor (with float dtype), breaking the mask.  This subclass bypasses
    that conversion and only adds *x* and *y* as differentiable parents.
    """

    def __call__(self, condition, x, y):
        """Wrap only x and y; keep condition as a raw array."""
        from .tensor import Tensor

        # Strip Tensor wrappers from condition so it stays as a bool array.
        if isinstance(condition, Tensor):
            condition = condition.data

        # Promote x / y from ndarray → Tensor if needed.
        if hasattr(x, "shape") and not isinstance(x, Tensor):
            x = Tensor(x)
        if hasattr(y, "shape") and not isinstance(y, Tensor):
            y = Tensor(y)

        return self._raw_call(condition, x, y)

    def _raw_call(self, condition, x, y):
        from .tensor import Tensor
        from .backend import get_xp

        x_data = x.data if isinstance(x, Tensor) else x
        y_data = y.data if isinstance(y, Tensor) else y
        xp = get_xp(x_data) if hasattr(x_data, "shape") else np
        out_data = xp.where(condition, x_data, y_data)

        parents = [p for p in (x, y) if isinstance(p, Tensor)]
        if not parents:
            return out_data

        return Tensor(
            out_data,
            parents=parents,
            op_func=self,
            op_kwargs={
                "condition": condition,
                "x_is_tensor": isinstance(x, Tensor),
                "y_is_tensor": isinstance(y, Tensor),
            },
            name=self.name,
        )

    def vpj(self, grad, *args, **op_kwargs):
        from .backend import get_xp

        xp = get_xp(grad)
        condition = op_kwargs["condition"]
        zeros = xp.zeros_like(grad)
        grads = []
        if op_kwargs.get("x_is_tensor", True):
            grads.append(xp.where(condition, grad, zeros))
        if op_kwargs.get("y_is_tensor", True):
            grads.append(xp.where(condition, zeros, grad))
        return tuple(grads)


where = _WhereOps(np.where, name="where")

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

# Embedding gather — differentiable index lookup (scatter-add VJP)
gather = _GatherOps(_gather_forward, name="gather")


# ---------------------------------------------------------------------------
# Array creation helpers — return Tensors so they are graph-compatible
# ---------------------------------------------------------------------------


def _tensor_creation(np_func, name):
    """Factory: wrap a numpy array-creation function to return a leaf Tensor.

    Uses the active backend (CuPy when available) so creation ops stay on the
    same device as the rest of the computation graph.
    """
    from .backend import backend as _backend

    _func_name = np_func.__name__

    def creator(*args, **kwargs):
        from .tensor import Tensor

        xp_func = getattr(_backend, _func_name, np_func)
        data = xp_func(*args, **kwargs)
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
    """Namespace for random array creation that returns Tensors.

    All methods use the active backend so random arrays land on the same
    device as the rest of the computation graph.
    """

    @staticmethod
    def randn(*shape):
        from .tensor import Tensor
        from .backend import backend as _backend

        return Tensor(_backend.random.randn(*shape), name="randn")

    @staticmethod
    def rand(*shape):
        from .tensor import Tensor
        from .backend import backend as _backend

        return Tensor(_backend.random.rand(*shape), name="rand")

    @staticmethod
    def randint(low, high=None, size=None):
        from .tensor import Tensor
        from .backend import backend as _backend, get_dtype

        return Tensor(
            _backend.random.randint(low, high=high, size=size).astype(get_dtype()),
            name="randint",
        )

    @staticmethod
    def uniform(low=0.0, high=1.0, size=None):
        from .tensor import Tensor
        from .backend import backend as _backend

        return Tensor(_backend.random.uniform(low, high, size), name="uniform")

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None):
        from .tensor import Tensor
        from .backend import backend as _backend

        return Tensor(_backend.random.normal(loc, scale, size), name="normal")

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
    # array manipulation
    "concatenate",
    "stack",
    "clip",
    "cumsum",
    "flip",
    "roll",
    "tile",
    "repeat",
    "where",
    "gather",
]
