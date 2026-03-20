"""
dnp/core/tensor.py
==================
`Tensor` — a differentiable array that participates in the dynamic
computation graph and supports reverse-mode automatic differentiation
via `backward()`.

Now uses composition to support both NumPy (CPU) and CuPy (GPU) backends.
"""

# Third-party libraries
import numpy as np

# Local imports
from .session import session
from .vjp_rules import VJP_RULES
from .backend import (
    get_xp,
    as_numpy,
    as_cupy,
    is_cupy_array,
    is_cuda_available,
    to_device,
    get_dtype,
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
    ):
        if device is None:
            # Default to GPU if the input is already a CuPy array, else CPU
            device = "cuda" if is_cupy_array(input_array) else "cpu"

        dtype = get_dtype()
        if device == "cuda":
            self.data = as_cupy(input_array)
            if self.data.dtype != dtype:
                self.data = self.data.astype(dtype)
        else:
            self.data = as_numpy(input_array)
            if self.data.dtype != dtype:
                self.data = self.data.astype(dtype)

        xp = get_xp(self.data)
        self.grad = xp.zeros_like(self.data, dtype=dtype)

        self.parents = parents if parents else []
        self.op_func = op_func
        self.op_kwargs = op_kwargs if op_kwargs is not None else {}
        self.name = name

        # Register this node in the session graph
        self.id = session.add_node(self)

        # Re-register any parent that was cleared by session.reset()
        for p in self.parents:
            if isinstance(p, Tensor):
                if getattr(p, "id", None) is None or p.id not in session._nodes:
                    p.id = session.add_node(p)
                session.add_edge(p.id, self.id)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        from .ops import transpose

        return transpose(self)

    @property
    def device(self):
        return "cuda" if get_xp(self.data).__name__ == "cupy" else "cpu"

    def cpu(self):
        """Move tensor data and gradients to CPU."""
        if self.device == "cuda":
            self.data = as_numpy(self.data)
            self.grad = as_numpy(self.grad)
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
            self.grad = as_cupy(self.grad)
        return self

    def to(self, device: str):
        """Move tensor to *device* ('cpu' or 'cuda')."""
        if device == "cuda":
            return self.cuda()
        return self.cpu()

    def backward(self, grad_entrant=None):
        """
        Propagate gradients back through the computation graph using an
        iterative topological sort (reverse-mode automatic differentiation).

        Why topological order instead of recursive DFS
        -----------------------------------------------
        In a graph where a node is used by *multiple* downstream ops
        (diamond / shared-weight pattern), the recursive approach would push
        a partial gradient through the node's parents on the *first* visit,
        before all downstream contributions have arrived.  The iterative
        topo-sort guarantees that every node has accumulated the *complete*
        incoming gradient before its VJP is evaluated — which is the
        mathematically correct result.  It also avoids Python's recursion
        limit on deep networks.
        """
        xp = get_xp(self.data)
        if grad_entrant is None:
            grad_entrant = xp.ones_like(self.data)

        # --- Step 1: Build reverse-topological order via iterative post-order DFS ---
        # We walk the *parent* edges (the reverse of the forward graph).
        # Post-order DFS gives leaves first; reversing gives root (self) first.
        topo: list = []
        visited: set = set()
        stack = [(self, False)]  # (node, already_expanded)
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

        topo.reverse()  # root (loss) first → leaves last

        # --- Step 2: Seed the root gradient ---
        self.grad += grad_entrant

        # --- Step 3: Propagate in topological order ---
        # Each node is visited exactly once, with its fully-accumulated gradient.
        for node in topo:
            if node.op_func not in VJP_RULES or not node.parents:
                continue

            if "_vjp_args" in node.op_kwargs:
                # Mixed scalar + Tensor positional args: use stored full arg list
                # so the VJP lambda receives the correct signature.
                # Wrap plain Python scalars as 0-d numpy arrays so VJP lambdas
                # that call `.shape` on all args (e.g. unbroadcast) don't fail.
                import numpy as _np_bwd

                vjp_args = [
                    _np_bwd.asarray(a) if isinstance(a, (int, float)) else a
                    for a in node.op_kwargs["_vjp_args"]
                ]
                parent_indices = node.op_kwargs["_vjp_parent_indices"]
                other_kwargs = {
                    k: v
                    for k, v in node.op_kwargs.items()
                    if k not in ("_vjp_args", "_vjp_parent_indices")
                }
                all_grads = VJP_RULES[node.op_func](
                    node.grad, *vjp_args, **other_kwargs
                )
                # Only assign gradients for the Tensor parents (by original index)
                for parent, orig_idx in zip(node.parents, parent_indices):
                    if isinstance(parent, Tensor):
                        parent.grad += all_grads[orig_idx]
            else:
                args_data = [
                    p.data if isinstance(p, Tensor) else p for p in node.parents
                ]
                gradients = VJP_RULES[node.op_func](
                    node.grad, *args_data, **node.op_kwargs
                )
                for parent, g in zip(node.parents, gradients):
                    if isinstance(parent, Tensor):
                        parent.grad += g

    def __repr__(self):
        return f"Tensor({repr(self.data)}, name='{self.name}', device='{self.device}')"

    def __array__(self, dtype=None):
        return as_numpy(self.data).__array__(dtype)

    def reshape(self, *shape):
        from .ops import reshape

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, newshape=shape)

    def argmax(self, axis=None):
        xp = get_xp(self.data)
        return xp.argmax(self.data, axis=axis)

    def item(self):
        if self.size == 1:
            return self.data.item()
        raise ValueError("can only convert an array of size 1 to a Python scalar")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a simple sub-array. Gradient tracking for slices would require a specialized op.
        return self.data[idx]

    def __setitem__(self, idx, value):
        """In-place update of underlying data — used by optimizers (e.g. p[...] = p - lr * grad)."""
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value

    # --- Magic Methods for Operations ---
    def __add__(self, other):
        from .ops import add

        return add(self, other)

    def __radd__(self, other):
        from .ops import add

        return add(other, self)

    def __sub__(self, other):
        from .ops import subtract

        return subtract(self, other)

    def __rsub__(self, other):
        from .ops import subtract

        return subtract(other, self)

    def __mul__(self, other):
        from .ops import multiply

        return multiply(self, other)

    def __rmul__(self, other):
        from .ops import multiply

        return multiply(other, self)

    def __truediv__(self, other):
        from .ops import divide

        return divide(self, other)

    def __rtruediv__(self, other):
        from .ops import divide

        return divide(other, self)

    def __pow__(self, other):
        from .ops import power

        return power(self, other)

    def __rpow__(self, other):
        from .ops import power

        return power(other, self)

    def __matmul__(self, other):
        from .ops import matmul

        return matmul(self, other)

    def __rmatmul__(self, other):
        from .ops import matmul

        return matmul(other, self)

    def __neg__(self):
        from .ops import negative

        return negative(self)

    # --- Comparison operators (return plain numpy bool arrays, not tracked) ---
    # These are intentionally non-differentiable (used for masks / conditions).

    def __eq__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data == other_data

    def __ne__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data != other_data

    def __lt__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data < other_data

    def __le__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data <= other_data

    def __gt__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data > other_data

    def __ge__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data >= other_data

    def __abs__(self):
        from .ops import absolute

        return absolute(self)

    # --- Detach: leaf copy with no gradient history ---

    def detach(self):
        """Return a new leaf Tensor that shares no gradient history with self.

        Useful for implementing stop-gradient, target networks, or returning
        a value without propagating through it during ``backward()``.

        Example
        -------
        >>> target = current_params.detach()   # no gradient through target
        """
        return Tensor(
            self.data.copy(),
            name=self.name + "_detached",
            device=self.device,
        )
