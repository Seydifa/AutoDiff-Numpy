"""
dnp/core/ops.py
===============
``Ops`` class + differentiable operation instances.

Design paradigm
---------------
1. VJP rules and forward functions belong in ``vjp_rules.py``.  Define the
   forward callable and register its backward there with ``@vjp_rule``.
2. Kernel call functions belong here — named ``<opname>_kernel_call`` — for
   ops that need custom graph-wiring: caching intermediate values (dropout
   mask), or keeping non-differentiable arguments out of the graph (where's
   condition, gather's integer indices).
3. ``Ops`` instances are built here *after* both VJP rules and kernel calls
   are defined.  ``Ops(func, vjp_fn, name, kernel_call=None)`` binds the VJP
   once at construction — no runtime dict lookup — so a custom op registered
   under the same name always uses its own correct VJP.
4. ``tensor.py`` calls ``op.vjp(...)``; ``Ops.vjp`` delegates directly to
   ``self._vjp`` and never routes through ``VJP_RULES``.

Sections
--------
1. Imports
2. ``Ops`` base class
3. Kernel call functions  (where / gather / dropout)
4. Auto-generation loop  (``VJP_RULES`` → ``Ops`` instances, public names)
5. Manual wrappers for private-named helpers  (_reshape / _concatenate / _stack)
6. Special-case ops with kernel calls  (where / gather / embedding / dropout)
7. Array-creation helpers
8. ``_RandomModule``
9. ``__all__``
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from .tensor import Tensor  # graph node
from .backend import backend, safe_eps, get_dtype, is_cupy_array
from .session import graph_fn
import dnp.core.vjp_rules as rules

# VJP registry (func → vjp_fn), populated by @vjp_rule in vjp_rules.py.
# Imported here for auto-generation and for callers that need direct access.
VJP_RULES = rules.VJP_RULES


# ===========================================================================
# Device-stats tracker
# ===========================================================================


class _DeviceStats:
    """Records how many forward-pass operations ran on each device.

    Maintains two counters — ``'cpu'`` and ``'cuda'`` — that are incremented
    on every ``Ops.__call__``.  Use :func:`device_op_percentage` and
    :func:`reset_device_stats` from ``dnp.utils`` to query / reset the counts.
    """

    def __init__(self):
        self.counts: dict = {"cpu": 0, "cuda": 0}

    def record(self, device: str) -> None:
        """Increment the counter for *device* by one."""
        self.counts[device] = self.counts.get(device, 0) + 1

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.counts = {"cpu": 0, "cuda": 0}

    def total(self) -> int:
        """Return the total number of operations recorded."""
        return sum(self.counts.values())

    def percentage(self, device: str) -> float:
        """Return the percentage of operations that ran on *device*.

        Parameters
        ----------
        device : {'cpu', 'cuda'}
            The target device.

        Returns
        -------
        float
            Value in ``[0.0, 100.0]``.  Returns ``0.0`` when no operations
            have been recorded yet.
        """
        total = self.total()
        if total == 0:
            return 0.0
        return 100.0 * self.counts.get(device, 0) / total


#: Module-level singleton used by all ``Ops`` instances.
device_stats = _DeviceStats()


# ===========================================================================
# 2. Ops base class
# ===========================================================================


class Ops:
    """Bundles a forward function and its bound VJP into a graph-compatible callable.

    Parameters
    ----------
    func : callable
        The raw forward function (operates on plain arrays, not Tensors).
    vjp_fn : callable, optional
        ``vjp_fn(grad, *fwd_args, **fwd_kwargs) → tuple[grads]``.
        Bound once at construction — never fetched from ``VJP_RULES`` at
        call time.  Falls back to ``VJP_RULES.get(func)`` at init if omitted,
        so existing code that only passes ``func`` continues to work.
    name : str, optional
        Display name; defaults to ``func.__name__``.
    kernel_call : callable, optional
        ``kernel_call(op, *args, **kwargs) → Tensor``
        Full replacement for ``_raw_call`` for ops that need custom graph
        wiring.  Receives the ``Ops`` instance as first arg so it can set
        ``op_func=op`` on the output Tensor node.
    """

    def __init__(self, func, vjp_fn=None, name=None, kernel_call=None):
        self.func = func
        self.name = name or getattr(func, "__name__", repr(func))
        # Bind the VJP once — look up from the registry only as a fallback.
        self._vjp = vjp_fn if vjp_fn is not None else VJP_RULES.get(func)
        # Use the provided kernel call or the default graph-wiring path.
        if kernel_call is not None:
            raw = lambda *a, **kw: kernel_call(self, *a, **kw)  # noqa: E731
        else:
            raw = self._raw_call
        self._graph_call = graph_fn(raw)

    # ------------------------------------------------------------------
    def _raw_call(self, *args, **kwargs):
        """Default graph-wiring: strip .data → forward → build Tensor node."""
        args_data = [a.data if isinstance(a, Tensor) else a for a in args]
        kwargs_data = {
            k: (v.data if isinstance(v, Tensor) else v) for k, v in kwargs.items()
        }

        out_data = self.func(*args_data, **kwargs_data)

        parent_indices = [i for i, a in enumerate(args) if isinstance(a, Tensor)]
        parents = [args[i] for i in parent_indices]
        parents.extend(v for v in kwargs.values() if isinstance(v, Tensor))
        op_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, Tensor)}

        # Mixed-arg path: store full arg list so the VJP can reconstruct it.
        if len(parent_indices) < len(args):
            op_kwargs["_vjp_args"] = args_data
            op_kwargs["_vjp_parent_indices"] = parent_indices

        if not parents:
            return out_data

        return Tensor(
            out_data,
            parents=parents,
            op_func=self,
            op_kwargs=op_kwargs,
            name=self.name,
        )

    # ------------------------------------------------------------------
    def vjp(self, grad, *args, **op_kwargs):
        """Call the VJP bound at init.  Never routes through VJP_RULES."""
        if self._vjp is None:
            raise KeyError(f"No VJP rule for '{self.name}'")
        if "_vjp_args" in op_kwargs:
            full_args = [
                backend.asarray(a) if isinstance(a, (int, float)) else a
                for a in op_kwargs["_vjp_args"]
            ]
            clean_kw = {
                k: v
                for k, v in op_kwargs.items()
                if k not in ("_vjp_args", "_vjp_parent_indices")
            }
            return self._vjp(grad, *full_args, **clean_kw)
        return self._vjp(grad, *args, **op_kwargs)

    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        result = self._graph_call(*args, **kwargs)
        # Determine the device from the output and update the global tracker.
        if isinstance(result, Tensor):
            _dev = "cuda" if is_cupy_array(result.data) else "cpu"
        elif hasattr(result, "dtype"):  # bare numpy / cupy array
            _dev = "cuda" if is_cupy_array(result) else "cpu"
        else:
            _dev = backend.device
        device_stats.record(_dev)
        return result

    def __repr__(self):
        return f"Ops('{self.name}')"


# ===========================================================================
# 3. Kernel call functions for special ops
#    Signature: kernel_call(op, *args, **kwargs) → Tensor | array
#    These replace _raw_call for ops that need custom graph wiring.
#    Defined *before* the Ops instances that use them.
# ===========================================================================

# ---- where_kernel_call ----------------------------------------------------


def where_kernel_call(op, condition, x, y):
    """where: condition is not differentiable and is kept out of the graph."""
    # Strip condition from Tensor tracking.
    if isinstance(condition, Tensor):
        condition = condition.data
    # Promote bare arrays to Tensor for uniform parent tracking.
    if hasattr(x, "shape") and not isinstance(x, Tensor):
        x = Tensor(x)
    if hasattr(y, "shape") and not isinstance(y, Tensor):
        y = Tensor(y)

    x_data = x.data if isinstance(x, Tensor) else x
    y_data = y.data if isinstance(y, Tensor) else y
    out_data = backend.where(condition, x_data, y_data)

    parents = [p for p in (x, y) if isinstance(p, Tensor)]
    if not parents:
        return out_data
    return Tensor(
        out_data,
        parents=parents,
        op_func=op,
        op_kwargs={
            "condition": condition,
            "x_is_tensor": isinstance(x, Tensor),
            "y_is_tensor": isinstance(y, Tensor),
        },
        name=op.name,
    )


def _where_vjp(grad, *args, **op_kwargs):
    """VJP for the where Ops — uses condition/x_is_tensor/y_is_tensor from op_kwargs.

    Parents are only the tracked Tensor arguments (x and/or y); condition is
    never tracked.  The returned tuple has one entry per parent, in order.
    """
    condition = op_kwargs["condition"]
    zeros = backend.zeros_like(grad)
    grads = []
    if op_kwargs.get("x_is_tensor", True):
        grads.append(backend.where(condition, grad, zeros))
    if op_kwargs.get("y_is_tensor", True):
        grads.append(backend.where(condition, zeros, grad))
    return tuple(grads)


# ---- gather_kernel_call ---------------------------------------------------


def gather_kernel_call(op, table, idx):
    """gather/embedding: integer idx is not differentiable; kept out of the graph."""
    # Strip idx from Tensor tracking — indices are not differentiable.
    if isinstance(idx, Tensor):
        idx = idx.data
    idx = backend.asarray(idx, dtype=int)
    table_data = table.data if isinstance(table, Tensor) else table
    out_data = table_data[idx]
    parents = [table] if isinstance(table, Tensor) else []
    if not parents:
        return out_data
    return Tensor(
        out_data,
        parents=parents,
        op_func=op,
        op_kwargs={"idx": idx},
        name=op.name,
    )


# ---- dropout_kernel_call --------------------------------------------------


def dropout_kernel_call(op, x, p=0.5, training=True):
    """dropout: generates and caches the binary mask for the backward pass."""
    x_data = x.data if isinstance(x, Tensor) else x
    if training and p > 0:
        mask = (backend.random.uniform(size=x_data.shape) >= p).astype(x_data.dtype)
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
        op_func=op,
        op_kwargs={"p": p, "training": training, "mask": mask},
        name=op.name,
    )


# ===========================================================================
# 4. Auto-generate Ops for every public function registered in VJP_RULES
#    Public = name does not start with "_"
#    vjp_fn is passed explicitly so each Ops instance owns its VJP at init.
# ===========================================================================

for _fn, _vjp_fn in VJP_RULES.items():
    _n = getattr(_fn, "__name__", None)
    if _n and not _n.startswith("_"):
        globals()[_n] = Ops(_fn, vjp_fn=_vjp_fn, name=_n)

del _fn, _vjp_fn, _n  # clean up loop variables


# ===========================================================================
# 5. Manual wrappers for private-named helpers
#    These are public ops but their forward fn has a leading underscore.
# ===========================================================================

reshape = Ops(rules._reshape, vjp_fn=VJP_RULES[rules._reshape], name="reshape")
concatenate = Ops(
    rules._concatenate, vjp_fn=VJP_RULES[rules._concatenate], name="concatenate"
)
stack = Ops(rules._stack, vjp_fn=VJP_RULES[rules._stack], name="stack")


# ===========================================================================
# 6. Special-case ops with kernel calls
#    Explicit construction replaces the type() dynamic-class pattern.
#    Order: check VJP registration in vjp_rules.py, then define kernel call
#    above (Section 3), then build the Ops instance here.
# ===========================================================================

where = Ops(
    backend.where,
    vjp_fn=_where_vjp,  # ops-layer VJP — uses stored condition/flags
    name="where",
    kernel_call=where_kernel_call,
)

gather = Ops(
    rules.gather,
    vjp_fn=VJP_RULES[rules.gather],
    name="gather",
    kernel_call=gather_kernel_call,
)

embedding = Ops(
    rules.embedding,
    vjp_fn=VJP_RULES[rules.embedding],
    name="embedding",
    kernel_call=gather_kernel_call,
)

embedding_lookup = Ops(
    rules.embedding_lookup,
    vjp_fn=VJP_RULES[rules.embedding_lookup],
    name="embedding_lookup",
    kernel_call=gather_kernel_call,
)

dropout = Ops(
    rules.dropout,
    vjp_fn=VJP_RULES[rules.dropout],
    name="dropout",
    kernel_call=dropout_kernel_call,
)


# ===========================================================================
# 7. Array-creation helpers — return leaf Tensors
# ===========================================================================


def _tensor_creation(func_name, name):
    """Factory: call backend.<func_name> and return a leaf Tensor."""

    def creator(*args, **kwargs):
        return Tensor(getattr(backend, func_name)(*args, **kwargs), name=name)

    creator.__name__ = name
    creator.__qualname__ = name
    return creator


arange = _tensor_creation("arange", "arange")
linspace = _tensor_creation("linspace", "linspace")
ones = _tensor_creation("ones", "ones")
zeros = _tensor_creation("zeros", "zeros")
full = _tensor_creation("full", "full")
eye = _tensor_creation("eye", "eye")


def ones_like(x):
    """Tensor of ones matching x's shape and device."""
    src = x.data if isinstance(x, Tensor) else x
    return Tensor(backend.ones_like(src), name="ones_like")


def zeros_like(x):
    """Tensor of zeros matching x's shape and device."""
    src = x.data if isinstance(x, Tensor) else x
    return Tensor(backend.zeros_like(src), name="zeros_like")


def full_like(x, fill_value):
    """Tensor filled with fill_value matching x's shape and device."""
    src = x.data if isinstance(x, Tensor) else x
    return Tensor(backend.full_like(src, fill_value), name="full_like")


# ===========================================================================
# 8. _RandomModule — ops.random.*
# ===========================================================================


class _RandomModule:
    """backend-agnostic random Tensor factory (ops.random.randn, etc.)."""

    @staticmethod
    def randn(*shape):
        return Tensor(backend.random.randn(*shape), name="randn")

    @staticmethod
    def rand(*shape):
        return Tensor(backend.random.rand(*shape), name="rand")

    @staticmethod
    def randint(low, high=None, size=None):
        return Tensor(
            backend.random.randint(low, high=high, size=size).astype(get_dtype()),
            name="randint",
        )

    @staticmethod
    def uniform(low=0.0, high=1.0, size=None):
        return Tensor(backend.random.uniform(low, high, size), name="uniform")

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None):
        return Tensor(backend.random.normal(loc, scale, size), name="normal")

    @staticmethod
    def seed(s):
        backend.random.seed(s)


random = _RandomModule()


# ===========================================================================
# 9. __all__
# ===========================================================================

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
    # trig & hyperbolic
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
    # nn activations
    "sigmoid",
    "relu",
    "leaky_relu",
    "elu",
    "softplus",
    "swish",
    "gelu",
    "softmax",
    # conv
    "conv2d",
    # pooling
    "max_pool2d",
    "avg_pool2d",
    # regularization / normalization
    "dropout",
    "batch_norm",
    # gather / embedding
    "gather",
    "embedding",
    "embedding_lookup",
    # FFT
    "fft",
    "ifft",
    "fftn",
    "ifftn",
    # recurrent & sequence
    "rnn_cell",
    "lstm_cell",
    "gru_cell",
    "layer_norm",
    "scaled_dot_product_attention",
    # loss functions
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "log_cosh_loss",
    "bce_loss",
    "bce_with_logits_loss",
    "cce_loss",
    "cce_with_logits_loss",
    "sparse_cce_with_logits_loss",
    "nll_loss",
    "kl_divergence_loss",
    "focal_loss",
    "hinge_loss",
    "squared_hinge_loss",
    "cosine_embedding_loss",
    "triplet_margin_loss",
    "dice_loss",
    "tversky_loss",
    "wasserstein_loss",
    "ssim_loss",
    # creation
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
    # device tracking
    "device_stats",
]
