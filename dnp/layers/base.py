"""
Base module classes and utilities.
"""

from dnp.core.backend import backend, get_dtype
from dnp.core.tensor import Tensor


def _apply_initializer(name: str, shape, dtype):
    """Resolve a built-in initializer name to an initial weight ndarray."""
    fan_in = shape[0] if len(shape) >= 1 else 1
    fan_out = shape[1] if len(shape) >= 2 else 1
    limit: float
    if name == "glorot_uniform":
        limit = float(backend.sqrt(6.0 / (fan_in + fan_out)))
        return backend.random.uniform(-limit, limit, shape).astype(dtype)
    if name == "glorot_normal":
        std = float(backend.sqrt(2.0 / (fan_in + fan_out)))
        return (backend.random.randn(*shape) * std).astype(dtype)
    if name == "he_normal":
        std = float(backend.sqrt(2.0 / fan_in))
        return (backend.random.randn(*shape) * std).astype(dtype)
    if name == "ones":
        return backend.ones(shape, dtype=dtype)
    if name == "zeros":
        return backend.zeros(shape, dtype=dtype)
    if name == "uniform":
        return backend.random.uniform(-0.05, 0.05, shape).astype(dtype)
    raise ValueError(
        f"Unknown initializer '{name}'. "
        "Choose from 'glorot_uniform', 'glorot_normal', 'he_normal', 'ones', 'zeros', 'uniform'."
    )


class Module:
    """Base class for all neural network modules."""

    _instance_counters: dict = {}

    def __init__(self):
        self.__dict__["_parameters"] = {}  # trainable weights
        self.__dict__["_nontrainable"] = {}  # non-trainable tensors
        self.__dict__["_buffers"] = {}  # plain arrays (not Tensors)
        self.__dict__["_modules"] = {}
        self.__dict__["training"] = True
        cls_name = self.__class__.__name__
        idx = Module._instance_counters.get(cls_name, -1) + 1
        Module._instance_counters[cls_name] = idx
        self.__dict__["_instance_name"] = f"{cls_name}_{idx}"

    def add_weight(
        self,
        name: str,
        shape,
        initializer="glorot_uniform",
        trainable: bool = True,
        dtype=None,
        **kwargs,
    ) -> "Tensor":
        _dtype = dtype or get_dtype()

        if callable(initializer) and not isinstance(initializer, str):
            # Evaluate callable and ensure it rests on the correct backend
            data = backend.asarray(initializer(shape)).astype(_dtype)
        else:
            data = _apply_initializer(initializer, shape, _dtype)

        w = Tensor(data, name=f"{self._instance_name}.{name}")
        object.__setattr__(self, name, w)
        if trainable:
            self._parameters[name] = w
        else:
            self._nontrainable[name] = w
        return w

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            if name in self._parameters:
                self._parameters[name] = value
            elif name in self._nontrainable:
                self._nontrainable[name] = value
            else:
                self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix=""):
        for name, param in self._parameters.items():
            full = f"{prefix}.{name}" if prefix else name
            yield full, param
        for mod_name, module in self._modules.items():
            sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            yield from module.named_parameters(sub_prefix)

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.fill(0.0)

    def cpu(self):
        for name, param in self._parameters.items():
            param.cpu()
        for name, module in self._modules.items():
            module.cpu()
        return self

    def cuda(self):
        for name, param in self._parameters.items():
            param.cuda()
        for name, module in self._modules.items():
            module.cuda()
        return self

    def to(self, device: str):
        """Move the module and all its parameters to *device* ('cpu' or 'cuda')."""
        return self.cuda() if device == "cuda" else self.cpu()

    def eval(self):
        self.training = False
        for module in self._modules.values():
            module.eval()
        return self

    def train(self):
        self.training = True
        for module in self._modules.values():
            module.train()
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "The forward() method must be implemented in subclasses."
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the total number of scalar parameters in this module.

        Parameters
        ----------
        trainable_only : bool
            When ``True`` (default) count only trainable parameters.
            Pass ``False`` to include non-trainable tensors as well.
        """
        params = dict(self._parameters)
        if not trainable_only:
            params.update(self._nontrainable)
        count = sum(p.size for p in params.values())
        for module in self._modules.values():
            count += module.num_parameters(trainable_only)
        return count

    def state_dict(self) -> dict:
        """Return an ordered dict mapping parameter names to numpy arrays."""
        from dnp.core.backend import as_numpy

        return {
            name: as_numpy(param.data).copy() for name, param in self.named_parameters()
        }

    def load_state_dict(self, state: dict) -> None:
        """Load parameter values from *state* (a dict from :meth:`state_dict`).

        Raises ``KeyError`` if a parameter name present in the model is missing
        from *state*, preventing silent shape mismatches.
        """
        for name, param in self.named_parameters():
            if name not in state:
                raise KeyError(
                    f"load_state_dict: missing key '{name}' in the provided state dict."
                )
            param.data[...] = state[name]

    @classmethod
    def reset_counters(cls) -> None:
        """Reset all module instance counters to zero.

        Useful in test suites to get deterministic module names across tests.
        """
        cls._instance_counters.clear()

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for name, module in self._modules.items():
            module_repr = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {module_repr}")
        lines.append(")")
        return "\n".join(lines)


class Sequential(Module):
    """Container that executes modules in sequence."""

    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
