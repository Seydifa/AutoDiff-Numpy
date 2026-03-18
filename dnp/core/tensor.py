"""
dnp/core/tensor.py
==================
`Tensor` — a NumPy ndarray subclass that participates in the dynamic
computation graph and supports reverse-mode automatic differentiation
via `backward()`.

Previously embedded inside dnp/autograd/session.py.
"""

# Third-party libraries
import numpy as np

# Local imports
from .session import session
from .vjp_rules import VJP_RULES


class Tensor(np.ndarray):
    """
    A differentiable array that records its creation operation and
    parent tensors into the global `session` computation graph.
    """

    def __new__(cls, input_array, parents=None, op_func=None, op_kwargs=None, name="Var"):
        obj = np.asarray(input_array, dtype=np.float64).view(cls)
        obj.grad = np.zeros_like(obj)
        obj.parents = parents if parents else []
        obj.op_func = op_func
        obj.op_kwargs = op_kwargs if op_kwargs is not None else {}
        obj.name = name

        # Register this node in the session graph
        obj.id = session.add_node(obj)

        # Re-register any parent that was cleared by session.reset()
        for p in obj.parents:
            if getattr(p, "id", None) is None or not session.G.has_node(p.id):
                p.id = session.add_node(p)
            session.add_edge(p.id, obj.id)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.grad = getattr(obj, "grad", np.zeros(self.shape, dtype=np.float64))
        self.parents = getattr(obj, "parents", [])
        self.op_func = getattr(obj, "op_func", None)
        self.op_kwargs = getattr(obj, "op_kwargs", {})
        self.name = getattr(obj, "name", "View")
        self.id = getattr(obj, "id", None)

    def backward(self, grad_entrant=None):
        """
        Propagate gradients back through the computation graph
        (reverse-mode automatic differentiation).
        """
        if grad_entrant is None:
            grad_entrant = np.ones_like(self)

        self.grad += grad_entrant

        if self.op_func in VJP_RULES and self.parents:
            args_data = list(self.parents)
            kwargs_data = self.op_kwargs
            gradients_locaux = VJP_RULES[self.op_func](self.grad, *args_data, **kwargs_data)

            for parent, grad_local in zip(self.parents, gradients_locaux):
                if isinstance(parent, Tensor):
                    parent.backward(grad_local)
