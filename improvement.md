in our branch v3, we define 
# Graph organisation change
we need to add graph_fn wrapper to session.py or tensor.py
why ?, the goal of this convert all input into graph compatible:
how to use it :

`
@graph_fn
f(**arg)
`
with that all argument are converted into Tensor if not and if they data argumen

# Ops Change
we need make all ops call compatible to Graph
for that we wrap all call with graph_fn
```
x = np.array([1, 2, 3])
y = ops.sin(x)
```
x is converted into Tensor automatically before to pass it into ours function via graph_fn

# Tensor  change
optimize the Tensor class

# Module change 
we add to add tensor construction in module meaning, we need to rewrite a Module to be easy to use and secure.
```
class Layer(Module):
    def __init__(self, *args, **karwgs):
        super().__init__()
        self.add_weight(
            trainable = False or True 
            **kwargs
        )
```

# Extend ops
We need to wrap numpy or cupy array construction for arange, ones, ones_like, zeros, zeros_like, random module and so on, to return tensor. And be more compatible for graph.

# Utility
We build utility for training, like Trainer class, Callbacks, Saving and so on

---

# Additional Improvements Found in v2 Codebase

## Critical Bugs

### 1. `dropout` VJP is incorrect — mask is not saved
`vjp_rules.py` — the backward for dropout divides by `(1-p)` uniformly,
but the actual binary mask used in the forward pass is discarded and never
stored. The correct backward must re-apply the same mask.
```python
# current (wrong)
dropout: lambda g, x, p=0.5, training=True: (g/(1-p),) if training else (g,),

# correct: mask must be saved during forward and reused in backward
# e.g. store mask on the Tensor as op_kwargs["mask"]
```

### 2. `conv2d` (scipy) VJP has an Ellipsis bug
In `VJP_RULES[conv2d]`, the weight gradient when mode != "valid" passes Python's
`...` (Ellipsis object) as the first argument to `conv2d`, which will crash:
```python
conv2d(x if mode=="valid" else ..., g, mode="valid"),  # BUG: ... is not an array
```
The full correlation backward for non-"valid" modes needs a proper implementation.

### 3. `_batch_norm_backward` divides by wrong `N` for `BatchNorm2d`
`N = x.shape[0]` is only correct for 1D batch norm (axis=0).
For `BatchNorm2d`, normalization is over axes `(0, 2, 3)`, so
`N = x.shape[0] * x.shape[2] * x.shape[3]`.

### 4. `np.sum` / `np.mean` / `np.prod` VJPs fail with tuple `axis`
`get_xp(g).expand_dims(g, axis)` does not accept a tuple axis in NumPy < 2.0.
Reductions like `mean(x, axis=(0,2,3))` (used in BatchNorm2d) will raise a
`TypeError`. Need to loop over axes or use `np.reshape` to restore dims.

---

## Architecture / Design Issues

### 5. Session graph grows forever — memory leak during training
`session` is a global singleton. Every `Tensor` construction adds nodes and
edges that are never removed. After thousands of training steps, `session._nodes`
and `session._edges` become massive. Options:
- Auto-reset session at the start of each forward pass
- Add `session.reset()` call in `Optimizer.zero_grad()` or after `backward()`
- Support a per-forward-pass context manager: `with session.graph():  ...`

### 6. Optimizer `step()` runs inside the live computation graph
`p[...] = p - self.lr * p.grad` calls `__sub__`, `__mul__`, then `__setitem__`.
The subtraction creates a new Tensor and adds it to the session graph — this
pollutes the graph with optimizer arithmetic.
All optimizer `step()` methods should run under `with session.no_grad(): ...`.

### 7. `Tensor` is not exported from the top-level `dnp/__init__.py`
Users cannot do `import dnp; t = dnp.Tensor(...)`. They must know the internal
path. Add `Tensor` (and `session`) to `dnp/__init__.py`.

### 8. No `detach()` method on `Tensor`
There is no way to stop gradient flow through a tensor (create a leaf copy
with no parents). This is essential for implementing things like target networks,
stop-gradient in contrastive learning, or simply returning a value without
backpropagating through it.
```python
def detach(self):
    return Tensor(self.data.copy(), name=self.name + "_detached")
```

### 9. `Embedding` is O(vocab_size) — uses one-hot matrix multiply
```python
one_hot = np.eye(self.num_embeddings, dtype=np.float32)[x_np]
return self.lin(Tensor(one_hot, ...))
```
This allocates a `(seq_len, vocab_size)` one-hot matrix and does a full matmul.
Replace with a direct index slice: `self.W.data[x_np]` and wire a custom
`gather` op with its VJP (scatter-add) for efficiency.

### 10. No learning rate scheduler
There is no `LRScheduler` base class. Essential schedulers missing:
- `StepLR`, `MultiStepLR`
- `CosineAnnealingLR`
- `ReduceLROnPlateau`
- `WarmupScheduler`

---

## Minor Issues

### 11. `LayerNorm` and `CrossEntropyLoss` add unnecessary Tensor nodes for epsilon
```python
eps_t = Tensor(np.array([self.eps]))   # in LayerNorm
eps_tensor = Tensor(np.array([1e-8]))  # in CrossEntropyLoss
```
These create leaf nodes in the session graph on every forward call.
Use scalar Python `float` addition instead — it won't be tracked because
non-Tensor scalars are not registered as parents.

### 12. `Ops.vpj_fun` is stored but never used
In `ops.py`, `self.vpj_fun` is set in `Ops.__init__` and exposed via `.vpj()`,
but `Tensor.backward()` always looks up the VJP through `VJP_RULES[node.op_func]`
directly — bypassing this field entirely. Remove or actually use it to make the
design coherent (e.g., let `Op.__call__` register `self.vpj_fun` into `VJP_RULES`).

### 13. `ops.py` shadows Python built-ins
`ops.sum`, `ops.max`, `ops.min`, `ops.round` shadow built-in Python functions.
Any code that does `from dnp.core.ops import *` or `from dnp import ops` will
lose access to the built-ins in that scope. Rename to e.g. `reduce_sum`,
`reduce_max`, etc., or keep the current names but make the shadowing explicit
in documentation.

### 14. `ScaledDotProductAttention` allocates a new Tensor for the scale factor on every call
```python
scores = scores * Tensor(np.array([self.scale], dtype=np.float32), device=query.device)
```
This adds a leaf node to the session graph on every attention forward pass.
Multiply by the Python float scalar `self.scale` directly instead.

### 15. `Tensor` missing comparison and unary operators
`__eq__`, `__ne__`, `__lt__`, `__gt__`, `__le__`, `__ge__`, `__abs__` are not
defined. `t1 > t2` currently falls back to numpy broadcasting and returns a
numpy array, not a Tensor, which breaks any downstream dnp operation that expects
a Tensor (e.g., masks in attention).

### 16. `VJP_RULES.md` in `dnp/ops/` is likely outdated
The VJP logic moved to `dnp/core/vjp_rules.py`, but the documentation file
lives in the old location. Either update or move it to `dnp/core/`.


### 17 Add loss module and implement 
Add a module for loss function, for automatic loss import and use and needed we can add vjp_rule for each loss for more robust solution  