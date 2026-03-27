"""
dnp/core/tensor.py
==================
`Tensor` — a differentiable array that participates in the dynamic
computation graph and supports reverse-mode automatic differentiation
via `backward()`.

Now uses composition to support both NumPy (CPU) and CuPy (GPU) backends.
"""

from .session import session
from .backend import (
    backend,
    get_dtype,
    as_numpy,
    as_cupy,
    is_cuda_available,
    is_cupy_array,
)


class Tensor:
    """
    A differentiable array that records its creation operation and
    parent tensors into the global `session` computation graph.
    """

    def __init__(
        self,
        input_array,
        parents=None,
        op_func=None,
        op_kwargs=None,
        name="Var",
        device=None,
        requires_grad=True,
    ):
        """
        Initialize a new Tensor.

        Parameters
        ----------
        input_array : array-like
            The raw data, automatically cast to the active precision and backend.
        parents : list of Tensor, optional
            Nodes that produced this Tensor through an operation.
        op_func : Ops, optional
            The graph-operation instance responsible for the `.vjp()` calculation.
        op_kwargs : dict, optional
            Any static kwargs or structural attributes required by `op_func`.
        name : str, default 'Var'
            Display name registered in the compute graph.
        device : {'cpu', 'cuda'}, optional
            Target device context. Defaults to inferring from `input_array` type.
        """
        if device is None:
            # If the input is already a CuPy array, keep it on GPU.
            # Otherwise respect whatever device the backend singleton is set to,
            # so that backend.set_device('cpu') / backend.set_device('cuda')
            # is honoured for all freshly-created leaf tensors.
            if is_cupy_array(input_array):
                device = "cuda"
            else:
                device = backend.device

        dtype = get_dtype()
        if device == "cuda":
            self.data = as_cupy(input_array)
        else:
            self.data = as_numpy(input_array)

        if "complex" not in str(self.data.dtype):
            if self.data.dtype != dtype:
                self.data = self.data.astype(dtype)

        # P1: lazy grad allocation — only created when accumulated during backward()
        self._grad = None

        self.requires_grad = requires_grad
        self.parents = parents if parents else []
        self.op_func = op_func
        self.op_kwargs = op_kwargs if op_kwargs is not None else {}
        self.name = name

        # Register this node in the session graph only when gradient tracking is on
        if self.requires_grad:
            self.id = session.add_node(self)

            # Re-register any parent that was cleared by session.reset()
            for p in self.parents:
                if isinstance(p, Tensor):
                    if getattr(p, "id", None) is None or p.id not in session._nodes:
                        p.id = session.add_node(p)
                    session.add_edge(p.id, self.id)
        else:
            self.id = None

    @property
    def grad(self):
        """Gradient buffer (None until backward() accumulates into this tensor)."""
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def shape(self):
        """Tuple of array dimensions."""
        return self.data.shape

    @property
    def ndim(self):
        """Number of array dimensions."""
        return self.data.ndim

    @property
    def dtype(self):
        """Data-type of the array's elements."""
        return self.data.dtype

    @property
    def size(self):
        """Number of elements in the array."""
        return self.data.size

    @property
    def T(self):
        """The transposed array."""
        return self._op("transpose", None)

    @property
    def device(self):
        """Hardware device context string ('cpu' or 'cuda')."""
        return "cuda" if is_cupy_array(self.data) else "cpu"

    def cpu(self):
        """Move tensor data and gradients to CPU."""
        if self.device == "cuda":
            self.data = as_numpy(self.data)
            if self._grad is not None:
                self._grad = as_numpy(self._grad)
        return self

    def cuda(self):
        """Move tensor data and gradients to GPU (CuPy)."""
        if not is_cuda_available:
            raise RuntimeError(
                "CuPy is not available. Install it for GPU support: "
                "https://docs.cupy.dev/en/stable/install.html"
            )
        if self.device == "cpu":
            self.data = as_cupy(self.data)
            if self._grad is not None:
                self._grad = as_cupy(self._grad)
        return self

    def to(self, device: str):
        """Move tensor to *device* ('cpu' or 'cuda')."""
        return self.cuda() if device == "cuda" else self.cpu()

    def zero_grad(self):
        """Reset the gradient buffer to zeros (allocates if not yet created)."""
        if self._grad is not None:
            self._grad.fill(0.0)
        # If grad was never allocated, leave it as None — backward will create it

    def backward(self, grad_entrant=None):
        """Propagate gradients back through the computation graph."""
        if grad_entrant is None:
            grad_entrant = backend.ones_like(self.data)

        # Build reverse-topological order via iterative post-order DFS
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            nid = id(node)
            if nid in visited:
                continue
            if expanded:
                topo.append(node)
                visited.add(nid)
            else:
                stack.append((node, True))
                for p in node.parents:
                    if isinstance(p, Tensor) and id(p) not in visited:
                        stack.append((p, False))

        topo.reverse()
        # Accumulate root gradient (lazy allocation)
        if self._grad is None:
            self._grad = (
                grad_entrant.copy() if hasattr(grad_entrant, "copy") else grad_entrant
            )
        else:
            self._grad = self._grad + grad_entrant

        # Propagate cleanly using the unified `.vjp` method on `op_func`
        for node in topo:
            op = node.op_func
            if op is None or not hasattr(op, "vjp") or not node.parents:
                continue
            g_node = node._grad
            if g_node is None:
                continue

            if "_vjp_args" in node.op_kwargs:
                # Dispatches via VJP reconstruction pattern
                parent_indices = node.op_kwargs["_vjp_parent_indices"]
                all_grads = op.vjp(g_node, **node.op_kwargs)
                for parent, orig_idx in zip(node.parents, parent_indices):
                    g = all_grads[orig_idx]
                    if isinstance(parent, Tensor) and g is not None:
                        if parent._grad is None:
                            parent._grad = g.copy() if hasattr(g, "copy") else g
                        else:
                            parent._grad = parent._grad + g
            else:
                # Dispatches raw VJP args normally
                args_data = [
                    p.data if isinstance(p, Tensor) else p for p in node.parents
                ]
                gradients = op.vjp(g_node, *args_data, **node.op_kwargs)
                for parent, g in zip(node.parents, gradients):
                    if isinstance(parent, Tensor) and g is not None:
                        if parent._grad is None:
                            parent._grad = g.copy() if hasattr(g, "copy") else g
                        else:
                            parent._grad = parent._grad + g

    def __repr__(self):
        return f"Tensor({repr(self.data)}, name='{self.name}', device='{self.device}')"

    def __float__(self):
        if self.size == 1:
            return float(as_numpy(self.data).flat[0])
        raise ValueError("only a size-1 Tensor can be converted to a Python float")

    def __int__(self):
        if self.size == 1:
            return int(as_numpy(self.data).flat[0])
        raise ValueError("only a size-1 Tensor can be converted to a Python int")

    def __bool__(self):
        if self.size == 1:
            return bool(as_numpy(self.data).flat[0])
        raise ValueError("only a size-1 Tensor can be converted to a Python bool")

    def __array__(self, dtype=None):
        return as_numpy(self.data).__array__(dtype)

    def reshape(self, *shape):
        """Gives a new shape to an array without changing its data."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return self._op("reshape", None, kwargs={"newshape": shape})

    def argmax(self, axis=None):
        """Returns the indices of the maximum values along an axis."""
        return backend.argmax(self.data, axis=axis)

    def item(self):
        """Copy an element of an array to a standard Python scalar and return it."""
        if self.size == 1:
            return self.data.item()
        raise ValueError("can only convert an array of size 1 to a Python scalar")

    def __len__(self):
        """Returns the length of the first dimension of the array."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a differentiable slice of this Tensor, tracked in the graph."""
        import dnp.core.ops as ops

        return ops.getitem(self, idx)

    def __setitem__(self, idx, value):
        """In-place update of underlying data — primarily used by optimizers.

        .. warning::
            This is a **non-differentiable** in-place mutation.  Gradient flow
            through ``x[idx] = value`` is *not* tracked.  For a differentiable
            alternative use ``ops.setitem(x, idx, value)`` which returns a new
            Tensor with the gradient intact for both ``x`` and ``value``.
        """
        self.data[idx] = value.data if isinstance(value, Tensor) else value

    # --- Differentiable Operations via internal `_op` logic ---
    def _op(self, name, other, reverse=False, kwargs=None):
        import dnp.core.ops as ops

        func = getattr(ops, name)
        kw = kwargs or {}
        if other is None:
            return func(self, **kw)
        return func(other, self, **kw) if reverse else func(self, other, **kw)

    def __add__(self, other):
        return self._op("add", other)

    def __radd__(self, other):
        return self._op("add", other, reverse=True)

    def __sub__(self, other):
        return self._op("subtract", other)

    def __rsub__(self, other):
        return self._op("subtract", other, reverse=True)

    def __mul__(self, other):
        return self._op("multiply", other)

    def __rmul__(self, other):
        return self._op("multiply", other, reverse=True)

    def __truediv__(self, other):
        return self._op("divide", other)

    def __rtruediv__(self, other):
        return self._op("divide", other, reverse=True)

    def __pow__(self, other):
        return self._op("power", other)

    def __rpow__(self, other):
        return self._op("power", other, reverse=True)

    def __matmul__(self, other):
        return self._op("matmul", other)

    def __rmatmul__(self, other):
        return self._op("matmul", other, reverse=True)

    def __neg__(self):
        return self._op("negative", None)

    def __abs__(self):
        return self._op("absolute", None)

    # --- Non-differentiable array operations (fall back to native ndarray magic) ---
    def _unwrap(self, other):
        return other.data if isinstance(other, Tensor) else other

    def __floordiv__(self, other):
        return self.data // self._unwrap(other)

    def __rfloordiv__(self, other):
        return self._unwrap(other) // self.data

    def __mod__(self, other):
        return self.data % self._unwrap(other)

    def __eq__(self, other):
        return self.data == self._unwrap(other)

    def __ne__(self, other):
        return self.data != self._unwrap(other)

    def __lt__(self, other):
        return self.data < self._unwrap(other)

    def __le__(self, other):
        return self.data <= self._unwrap(other)

    def __gt__(self, other):
        return self.data > self._unwrap(other)

    def __ge__(self, other):
        return self.data >= self._unwrap(other)

    def detach(self):
        """Return a new leaf Tensor sharing the underlying data buffer with no gradient history."""
        return Tensor(self.data, name=self.name + "_detached", device=self.device)
