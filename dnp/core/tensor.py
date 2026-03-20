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
from .backend import get_xp, as_numpy, as_cupy, is_cuda_available


class Tensor:
    """
    A differentiable array that records its creation operation and
    parent tensors into the global `session` computation graph.
    """

    def __init__(self, input_array, parents=None, op_func=None, op_kwargs=None, name="Var", device=None):
        if device is None:
            # Default to GPU if the input is already a CuPy array, else CPU
            if is_cuda_available and 'cupy' in str(type(input_array)):
                device = 'cuda'
            else:
                device = 'cpu'

        if device == 'cuda':
            self.data = as_cupy(input_array)
        else:
            self.data = as_numpy(input_array)
            if self.data.dtype != np.float64:
                self.data = self.data.astype(np.float64)

        xp = get_xp(self.data)
        self.grad = xp.zeros_like(self.data, dtype=xp.float64)
        
        self.parents = parents if parents else []
        self.op_func = op_func
        self.op_kwargs = op_kwargs if op_kwargs is not None else {}
        self.name = name

        # Register this node in the session graph
        self.id = session.add_node(self)

        # Re-register any parent that was cleared by session.reset()
        for p in self.parents:
            if getattr(p, "id", None) is None or not session.G.has_node(p.id):
                p.id = session.add_node(p)
            session.add_edge(p.id, self.id)

    @property
    def shape(self): return self.data.shape

    @property
    def ndim(self): return self.data.ndim

    @property
    def dtype(self): return self.data.dtype

    @property
    def size(self): return self.data.size

    @property
    def T(self):
        from .ops import transpose
        return transpose(self)

    @property
    def device(self):
        return 'cuda' if get_xp(self.data).__name__ == 'cupy' else 'cpu'

    def cpu(self):
        """Move tensor data and gradients to CPU."""
        if self.device == 'cuda':
            self.data = as_numpy(self.data)
            self.grad = as_numpy(self.grad)
        return self

    def cuda(self):
        """Move tensor data and gradients to GPU (CuPy)."""
        if self.device == 'cpu':
            self.data = as_cupy(self.data)
            self.grad = as_cupy(self.grad)
        return self

    def backward(self, grad_entrant=None):
        """
        Propagate gradients back through the computation graph
        (reverse-mode automatic differentiation).
        """
        xp = get_xp(self.data)
        if grad_entrant is None:
            grad_entrant = xp.ones_like(self.data)

        self.grad += grad_entrant

        if self.op_func in VJP_RULES and self.parents:
            args_data = list(self.parents)
            kwargs_data = self.op_kwargs
            gradients_locaux = VJP_RULES[self.op_func](self.grad, *args_data, **kwargs_data)

            for parent, grad_local in zip(self.parents, gradients_locaux):
                if isinstance(parent, Tensor):
                    parent.backward(grad_local)

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
