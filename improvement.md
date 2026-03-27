# DifferentialNumpy (dnp) — Comprehensive Improvement Plan

> Full codebase review for the `v3` branch.  
> Each item is tagged by **severity** (Critical / Major / Minor) and **category**.

---

## Table of Contents

1. [Critical Bugs](#1-critical-bugs)
2. [Major Bugs](#2-major-bugs)
3. [Minor Bugs](#3-minor-bugs)
4. [Security](#4-security)
5. [Performance](#5-performance)
6. [Design / Architecture](#6-design--architecture)
7. [Missing Features](#7-missing-features)
8. [Code Quality](#8-code-quality)
9. [Testing Gaps](#9-testing-gaps)
10. [Compatibility](#10-compatibility)
11. [Summary Matrix](#11-summary-matrix)

---

## 1. Critical Bugs

### B1 — `RotaryPositionalEncoding.forward` signature mismatch (CRASH)
**File:** `dnp/layers/attention.py`  
The layer calls `ops.rope(x, dim=self.dim, seq_dim=seq_dim, base=self.base)`, but the underlying `rope(x, cos_freqs, sin_freqs)` in `dnp/core/vjp_rules.py` expects precomputed frequency tensors, not `dim`/`base` kwargs.  
**Impact:** Instant `TypeError` on any forward call.  
**Fix:** Either compute `cos_freqs`/`sin_freqs` inside the layer's `forward()` from `dim`/`base`, or change the `rope` function signature to accept `dim` and `base` and compute freqs internally.

### B2 — `SinkhornTransport.forward` passes arguments in wrong order (CRASH)
**File:** `dnp/layers/advanced.py`  
Calls `ops.sinkhorn(cost_matrix, r, c, epsilon=..., num_iters=...)`, but `sinkhorn(a, b, M, reg, num_iters)` in `dnp/core/vjp_rules.py` expects positional `(a, b, M, reg, num_iters)`.  
**Impact:** Wrong computation or crash.  
**Fix:** Align call order with the `sinkhorn` forward function signature.

### B3 — `NeuralODE.forward` is entirely non-functional (CRASH)
**File:** `dnp/layers/advanced.py`  
Calls `ops.neural_ode_solve(self.odefunc, y0, self.t0, self.t1, self.steps)`, but `neural_ode_solve(z0, t_span, steps)` in `dnp/core/vjp_rules.py` expects `(z0, t_span, steps)` with a hardcoded `f(z) = -0.5 * z` placeholder. The user-supplied `odefunc` is silently ignored.  
**Impact:** Layer does nothing useful; the ODE function is never invoked.  
**Fix:** Pass the callable `odefunc` into `neural_ode_solve` and use it in the Euler integration loop and its VJP.

---

## 2. Major Bugs

### B4 — `leaky_relu` VJP ignores `alpha` parameter (WRONG GRADIENTS)
**File:** `dnp/core/vjp_rules.py`  
```python
@vjp_rule(func=leaky_relu)
def _vjp_leaky_relu(g, x):
    return (g * backend.where(x > 0, 1.0, 0.01),)  # BUG: hardcoded 0.01
```
The forward `leaky_relu(x, alpha=0.01)` accepts custom alpha, but the backward always uses `0.01`. Any `LeakyReLU(alpha=0.2)` produces **incorrect gradients**.  
**Fix:** Accept `alpha=0.01` kwarg in VJP and use it in the `where`.

### B5 — `elu` VJP ignores `alpha` parameter (WRONG GRADIENTS)
**File:** `dnp/core/vjp_rules.py`  
Same issue as B4. The backward uses `1.0 * backend.exp(x)` instead of `alpha * backend.exp(x)`.  
**Fix:** Accept `alpha=1.0` kwarg in VJP and multiply by it.

### B6 — `Ops` instances bind to import-time backend, breaking device switching
**File:** `dnp/core/ops.py`  
All `Ops` instances (e.g., `add = Ops(backend.add, ...)`) capture `backend.add` at import time. If `backend.set_device('cuda')` is called later, the already-created `Ops` still hold the NumPy function reference.  
**Impact:** After calling `dnp.set_device('cuda')`, all ops still run on CPU.  
**Fix:** Make `Ops.func` resolve lazily through the `backend` proxy, or re-create ops on device switch. Alternatively, since `BackendWrapper` already delegates via `__getattr__`, store the attribute *name* and look it up at call time.

### B7 — `Tensor.__getitem__` drops out of the computational graph
**File:** `dnp/core/tensor.py`  
`__getitem__` returns raw `self.data[idx]`, not a `Tensor`. Slicing/indexing produces plain arrays that lose gradient tracking.  
**Impact:** Any model that uses tensor slicing (e.g., splitting attention heads) will silently lose gradients.  
**Fix:** Return a `Tensor` with `parents=[self]` and register a `getitem` VJP rule.

### B8 — `Module.zero_grad()` crashes when `grad` is `None`
**File:** `dnp/layers/base.py`  
`p.grad.fill(0.0)` will raise `AttributeError` if a parameter's `grad` is `None` (e.g., created under `no_grad()`). The `Optimizer.zero_grad()` in `dnp/core/optimizers.py` checks for `None`, but `Module.zero_grad()` does not.  
**Fix:** Add `if p.grad is not None:` guard before `p.grad.fill(0.0)`.

### B9 — `BatchNorm2d` training/eval gradient shape mismatch risk
**File:** `dnp/layers/normalization.py`  
In eval mode, `BatchNorm2d` reshapes `mean`/`var`/`weight`/`bias` to `(1, C, 1, 1)` then calls `ops.batch_norm`. The `_batch_norm_backward` uses `x.ndim == 4` to decide reduction axes. Reshaping intermediates may cause `unbroadcast` to produce shape mismatches.  
**Fix:** Ensure training and eval mode pass identically shaped tensors, or handle both cases in the VJP.

### B10 — `Conv1d` duplicate import in `dnp/core/__init__.py`
**File:** `dnp/core/__init__.py`  
`Conv1d` is listed twice in the import list.  
**Fix:** Remove the duplicate.

---

## 3. Minor Bugs

### B11 — `Module._instance_counters` is shared mutable class state
**File:** `dnp/layers/base.py`  
`_instance_counters: dict = {}` is shared across all subclasses and persists across tests, causing non-deterministic instance names.  
**Fix:** Provide a `Module.reset_counters()` classmethod, or scope counters to model instances.

### B12 — `Softmax` layer doesn't accept `axis` parameter
**File:** `dnp/layers/activations.py`  
`Softmax` always uses `axis=-1` with no way to configure it.  
**Fix:** Accept `dim` or `axis` in `__init__` (like PyTorch's `Softmax(dim=...)`).

### B13 — `S4Layer` has no learnable parameters
**File:** `dnp/layers/advanced.py`  
Takes `(u, B, C, log_dt, delta_A)` as forward args but none are registered as module weights. Unusable as a standalone `Module`.  
**Fix:** Register `B`, `C`, `log_dt`, etc. as parameters via `add_weight()`.

### B14 — `set_device` free function may be missing or broken
**File:** `dnp/__init__.py` / `dnp/core/backend.py`  
`dnp/__init__.py` imports `set_device` from `backend.py`, but `backend.py` may only expose it as `backend.set_device()` (a method). Calling `dnp.set_device('cuda')` at the top level may fail.  
**Fix:** Verify the free function exists in `backend.py` and delegates to `backend.set_device()`.

---

## 4. Security

### S1 — Pickle deserialization allows arbitrary code execution (CRITICAL)
**File:** `dnp/utils/trainer.py`  
`_load_model` uses `pickle.load(fh)` on user-provided file paths. This is CWE-502 (Deserialization of Untrusted Data).  
**Impact:** Loading a maliciously-crafted checkpoint executes arbitrary code.  
**Fix:** Switch to `numpy.savez` / `numpy.load` for weights-only serialization, or adopt `safetensors`. If pickle is kept, add a prominent security warning in the docstring and docs.

### S2 — No input validation on Tensor shapes
**File:** `dnp/core/tensor.py`  
Users can pass arbitrary shapes/types to layers with no early validation, producing cryptic errors deep in computation.  
**Fix:** Add shape validation in layer `__init__` and `forward` methods for catch-early errors.

---

## 5. Performance

### P1 — `Tensor.__init__` always allocates a gradient buffer
**File:** `dnp/core/tensor.py`  
`self.grad = backend.zeros_like(self.data)` allocates memory for every intermediate tensor, even under `no_grad()` or for tensors that never participate in backward.  
**Impact:** Doubles memory usage for all transient tensors.  
**Fix:** Lazy-allocate `grad` on first access or on `backward()`, or skip allocation when `session._grad_enabled is False`.

### P2 — RNN/LSTM/GRU backward re-runs the full forward pass
**File:** `dnp/core/vjp_rules.py`  
VJPs for `rnn_cell`, `lstm_cell`, `gru_cell` call `_rnn_forward_cache(...)` which re-runs the entire forward pass to rebuild caches, doubling computation.  
**Fix:** Cache forward activations during the forward pass and store them in `op_kwargs`.

### P3 — `_avg_pool2d_backward` has Python loops over kernel offsets
**File:** `dnp/core/vjp_rules.py`  
Nested `for kh in range(kH): for kw in range(kW):` — for a 7×7 kernel that's 49 Python iterations.  
**Fix:** Fully vectorize like `_max_pool2d_backward`.

### P4 — `_restore_reduced_dims` always forces `.copy()`
**File:** `dnp/core/vjp_rules.py`  
`backend.broadcast_to(result, x_shape).copy()` forces a full allocation every time.  
**Fix:** Only copy when the result is actually used in an in-place `+=` (i.e., in `backward`). Consider making the copy optional.

### P5 — `Ops.__call__` records device stats on every operation
**File:** `dnp/core/ops.py`  
Every single op inspects the result device and updates `device_stats`, adding overhead to the hot path.  
**Fix:** Make device tracking opt-in via a flag (e.g., `device_stats.enabled = True`).

### P6 — `graph_fn` decorator overhead on every op call
**File:** `dnp/core/session.py`  
Every `_graph_call` wraps arguments through `_to_tensor`, checking `hasattr(v, "shape")` for each arg.  
**Fix:** Fast-path: if all args are already `Tensor`, skip the conversion loop.

### P7 — Optimizer state keyed by `id(param)` — fragile
**File:** `dnp/core/optimizers.py`  
All optimizers (Adam, AdamW, Momentum, etc.) use `id(p)` as dict key. If a parameter object is recreated (e.g., during BatchNorm running stats update), the old state is orphaned and moments restart from zero.  
**Fix:** Use a stable identifier (parameter name + index) or store state on the Tensor itself.

### P8 — `log_cosh_loss` allocates `backend.array(2.0)` on every call
**File:** `dnp/core/vjp_rules.py`  
`backend.log(backend.array(2.0, dtype=r.dtype))` creates a new GPU/CPU array per call.  
**Fix:** Precompute `_LOG2 = float(backend.log(2.0))` at module level.

---

## 6. Design / Architecture

### D1 — French/English mixing in API and error messages
**Files:** `dnp/core/session.py`  
- `self.compteur` → `self.counter`
- `"L'objet {type(noeud)} passé au graphe doit posséder un attribut 'name'."` → English
- `"Le nœud parent"` / `"Le nœud enfant"` → English
- `show_graphe` alias → remove or rename
- `"Graphe de Calcul (Passe Avant)"` → `"Computation Graph (Forward Pass)"`
- `"⚠️ Matplotlib n'est pas installé"` → English  

**Fix:** Standardize all user-facing strings and identifiers to English.

### D2 — Three backward-compatibility shim directories
**Directories:** `dnp/ops/`, `dnp/autograd/`, `dnp/core/layers.py`  
All three are pure re-export shims.  
**Fix:** If v2 compat is no longer needed, remove them. If still needed, document the deprecation timeline and emit `DeprecationWarning` on import.

### D3 — Triple `__init__.py` re-export maintenance burden
**Files:** `dnp/__init__.py`, `dnp/core/__init__.py`, `dnp/layers/__init__.py`  
Every new layer must be added in 3+ places.  
**Fix:** Have the top-level `__init__.py` import from a single canonical source. Use `__all__` from the subpackage.

### D4 — Loss functions are `Module` subclasses but some lack `__init__`
**File:** `dnp/layers/loss.py`  
`CrossEntropyLoss`, `NLLLoss` inherit `Module` but have no `__init__`. This accidentally works but creates instance counter side-effects.  
**Fix:** Add explicit `__init__` with `super().__init__()` to all loss classes.

### D5 — No `__repr__` for layers showing hyperparameters
**File:** `dnp/layers/base.py`  
`print(model)` only shows `_modules` children, not per-layer hyperparameters (in_features, out_features, kernel_size, etc.).  
**Fix:** Override `__repr__` in `Linear`, `Conv2d`, etc. to include key hyperparameters (like PyTorch).

### D6 — `Tensor.detach()` always forces a full data copy
**File:** `dnp/core/tensor.py`  
`self.data.copy()` allocates new memory. A cheaper detach could share the underlying buffer.  
**Fix:** `Tensor(self.data, name='detached')` — create a new graph-independent Tensor without copying data.

### D7 — `session.graph()` context manager doesn't reset on exit
**File:** `dnp/core/session.py`  
The `finally` block is `pass` — the graph stays alive after exit. The docstring says "the graph stays alive so backward() can traverse it after the block," but this means the memory-leak prevention (the original purpose) only works on *entry*, not exit.  
**Fix:** Document clearly that users must call `session.reset()` after `backward()`, or provide a `session.graph_and_backward()` helper.

---

## 7. Missing Features

### F1 — No gradient clipping utility
No `clip_grad_norm_` or `clip_grad_value_` — essential for training RNNs, transformers, and preventing exploding gradients.  
**Fix:** Add `dnp.utils.clip_grad_norm_(parameters, max_norm)` and `dnp.utils.clip_grad_value_(parameters, clip_value)`.

### F2 — No `Tensor.requires_grad` flag
Every Tensor always participates in the graph. There's no way to mark a Tensor as not needing gradients at creation time.  
**Fix:** Add `requires_grad=True` parameter to `Tensor.__init__`. Skip graph registration when `False`.

### F3 — No `model.to(device)` method
`Module.cpu()` and `Module.cuda()` exist but there's no unified `model.to('cuda')`.  
**Fix:** Add `Module.to(device)` dispatching to `cpu()` or `cuda()`.

### F4 — No `save_state_dict` / `load_state_dict`
The current save/load in `trainer.py` uses parameter index ordering, which breaks silently if architecture changes.  
**Fix:** Implement named `state_dict()` / `load_state_dict(dict)` on `Module`.

### F5 — No `TransformerDecoderLayer`
Only encoder layers exist. Cross-attention decoder layers are missing.  
**Fix:** Implement `TransformerDecoderLayer` with self-attention + cross-attention + FFN.

### F6 — No `MaxPool1d` / `AvgPool1d`
`Conv1d` exists but its pooling counterparts don't.  
**Fix:** Add `MaxPool1d` and `AvgPool1d` layers wrapping 1D pooling ops.

### F7 — No DataLoader / batching utility
Users must write their own mini-batch loops.  
**Fix:** Add a simple `DataLoader(dataset, batch_size, shuffle)` in `dnp/utils/`.

### F8 — No `model.num_parameters()` helper
Users must write `sum(p.size for p in model.parameters())` everywhere.  
**Fix:** Add `Module.num_parameters(trainable_only=True)` method.

### F9 — Missing layers in top-level `dnp/__init__.py` exports
`Conv1d`, `FlashAttention`, `RotaryPositionalEncoding`, `RNN`, `LSTM`, `GRU` and advanced layers are not exported.  
**Fix:** Add them to `dnp/__init__.py`.

### F10 — No `Tensor.__setitem__` with gradient support
Tensor `__setitem__` exists but doesn't track the operation in the graph, breaking gradient flow for in-place modifications.  
**Fix:** Implement a differentiable `__setitem__` or document the limitation.

---

## 8. Code Quality

### Q1 — `pyproject.toml` claims Python 3.8+ but code uses 3.10+ syntax
**File:** `pyproject.toml`  
Code uses `dict | None` (PEP 604, Python 3.10+) and `list[dict]` (PEP 585, Python 3.9+).  
**Fix:** Update to `python = "^3.10"`, or rewrite type hints to use `typing.Optional[dict]` / `typing.List[dict]`.

### Q2 — Tests import `networkx` unconditionally
**File:** `dnp/test/session_test.py`  
`import networkx as nx` at top level, but it's an optional dependency.  
**Fix:** Guard with `pytest.importorskip("networkx")` or `try/except`.

### Q3 — Examples use `sys.path.insert` hack
**Files:** All files in `examples/`  
**Fix:** Document `pip install -e .` or use relative imports. Remove `sys.path` manipulation.

### Q4 — `__all__` in `dnp/ops/__init__.py` is truncated
**File:** `dnp/ops/__init__.py`  
The `__all__` list is cut off at `"max"`, missing many ops.  
**Fix:** Complete the list or auto-generate from `dnp.core.ops.__all__`.

### Q5 — Dead constant `EPSILON = 1e-8`
**File:** `dnp/core/vjp_rules.py`  
Defined but replaced by `safe_eps()` everywhere.  
**Fix:** Remove or mark as deprecated.

### Q6 — `_GELU_COEFF` computed at import time
**File:** `dnp/core/vjp_rules.py`  
`float(backend.sqrt(2.0 / backend.pi))` — computed once at import. If backend changes device later, this is fine (it's a float), but it's fragile design.  
**Fix:** Use a literal constant `0.7978845608028654` directly.

### Q7 — Inconsistent `name` parameter handling in layers
**Files:** `dnp/layers/linear.py`, `dnp/layers/activations.py`  
Layers store `name` via `self.__dict__['name']`, bypassing `Module.__setattr__`. This can shadow the auto-generated `_instance_name`.  
**Fix:** Use a single consistent naming mechanism.

### Q8 — `float16_diagnostic.py` and `benchmark.py` are not pytest tests
**Files:** `dnp/test/float16_diagnostic.py`, `dnp/test/benchmark.py`  
Standalone scripts, not part of the test suite.  
**Fix:** Either convert to pytest tests or move to a `scripts/` directory.

---

## 9. Testing Gaps

### T1 — No tests for loss function VJPs
None of the 20 loss function VJP rules (MSE, BCE, CrossEntropy, Focal, Dice, etc.) are tested for gradient correctness.  
**Fix:** Add finite-difference gradient checks for all loss VJPs.

### T2 — No gradient correctness tests for RNN/LSTM/GRU
`layers_test.py` only checks that `grad is not None` and shape matches, not numerical correctness.  
**Fix:** Add finite-difference gradient checks.

### T3 — No tests for `Trainer`, callbacks, or schedulers
No test file covers `Trainer.fit()`, `Trainer.evaluate()`, `EarlyStopping`, `ModelCheckpoint`, or `LRScheduler` subclasses.  
**Fix:** Add a `test_trainer.py` with basic training loop tests.

### T4 — No tests for attention/transformer layers
`MultiHeadAttention`, `TransformerEncoderLayer`, `PositionalEncoding` are untested.  
**Fix:** Add shape/gradient tests.

### T5 — No tests for `where`, `gather`, `embedding` kernel calls
Special ops with custom `kernel_call` have no dedicated tests.  
**Fix:** Add forward/backward tests for each.

### T6 — No gradient correctness test for `Conv1d`
Only checks `grad is not None`, not numerical values.  
**Fix:** Add finite-difference check.

### T7 — No tests for optimizer state management
No tests verify Adam/AdamW moment estimates update correctly or that schedulers adjust LR properly.  
**Fix:** Add unit tests for each optimizer's state after N steps.

### T8 — No tests for CPU↔GPU device switching
All tests run CPU-only. The entire CuPy code path is untested.  
**Fix:** Add conditional GPU tests (skip if CuPy unavailable).

### T9 — No tests for `BatchNorm` eval-mode behavior
Tests only exercise training mode. Running-stat-based eval inference is untested.  
**Fix:** Test `model.eval()` → forward → check output uses running stats.

### T10 — No integration tests for full training pipelines
No test runs a model through multiple epochs and checks loss decreases.  
**Fix:** Add a small integration test (e.g., XOR problem converges in <100 epochs).

---

## 10. Compatibility

### C1 — Python version mismatch
**File:** `pyproject.toml`  
Claims `python = "^3.8"`, actual code requires `^3.10`.  
**Fix:** Update `pyproject.toml`.

### C2 — `backend.lib.stride_tricks.as_strided` may fail on older CuPy
**File:** `dnp/core/vjp_rules.py`  
CuPy didn't have `cupy.lib.stride_tricks` until version 10+.  
**Fix:** Add a version check or document minimum CuPy version in `pyproject.toml`.

### C3 — `backend.scipy.fft` requires SciPy ≥ 1.4
**File:** `dnp/core/ops.py`  
FFT ops rely on `scipy.fft`. The current `scipy >= 1.7` dep covers this, but it should be documented.  
**Fix:** Add a note in README or check at import time.

### C4 — NumPy 2.0 `reshape` keyword change
**File:** `dnp/core/vjp_rules.py`  
The `_reshape` wrapper handles this for the ops layer, but direct `backend.reshape(a, newshape=...)` calls elsewhere could break.  
**Fix:** Audit all reshape calls and route through `_reshape`.

---

## 11. Summary Matrix

| ID | Category | Severity | File(s) | Status |
|----|----------|----------|---------|--------|
| B1 | Bug | **Critical** | `layers/attention.py` | ✅ Done |
| B2 | Bug | **Critical** | `layers/advanced.py` | ✅ Done |
| B3 | Bug | **Critical** | `layers/advanced.py` | ✅ Done |
| B4 | Bug | **Major** | `core/vjp_rules.py` | ✅ Done |
| B5 | Bug | **Major** | `core/vjp_rules.py` | ✅ Done |
| B6 | Bug | **Major** | `core/ops.py` | ✅ Done |
| B7 | Bug | **Major** | `core/tensor.py` | ✅ Done |
| B8 | Bug | **Major** | `layers/base.py` | ✅ Done |
| B9 | Bug | **Major** | `layers/normalization.py` | ✅ Done (verified OK) |
| B10 | Bug | **Major** | `core/__init__.py` | ✅ Done |
| B11 | Bug | Minor | `layers/base.py` | ✅ Done |
| B12 | Bug | Minor | `layers/activations.py` | ✅ Done |
| B13 | Bug | Minor | `layers/advanced.py` | ✅ Done |
| B14 | Bug | Minor | `__init__.py` / `backend.py` | ✅ Done (verified OK) |
| S1 | Security | **Critical** | `utils/trainer.py` | ✅ Done |
| S2 | Security | Minor | `core/tensor.py` | ✅ Done |
| P1 | Perf | **Major** | `core/tensor.py` | ✅ Done |
| P2 | Perf | **Major** | `core/vjp_rules.py` | ✅ Done |
| P3 | Perf | Minor | `core/vjp_rules.py` | ✅ Done |
| P4 | Perf | Minor | `core/vjp_rules.py` | ✅ Done |
| P5 | Perf | Minor | `core/ops.py` | ✅ Done |
| P6 | Perf | Minor | `core/session.py` | ✅ Done |
| P7 | Perf | Minor | `core/optimizers.py` | ✅ Done |
| P8 | Perf | Minor | `core/vjp_rules.py` | ✅ Done |
| D1 | Design | Minor | `core/session.py` | ✅ Done |
| D2 | Design | Minor | `ops/`, `autograd/`, `core/layers.py` | ✅ Done |
| D3 | Design | Minor | `__init__.py` (×3) | ✅ Done |
| D4 | Design | Minor | `layers/loss.py` | ✅ Done |
| D5 | Design | Minor | `layers/base.py` | ✅ Done |
| D6 | Design | Minor | `core/tensor.py` | ✅ Done |
| D7 | Design | Minor | `core/session.py` | ✅ Done |
| F1 | Feature | **Major** | (new) | ✅ Done |
| F2 | Feature | **Major** | `core/tensor.py` | ✅ Done |
| F3 | Feature | Minor | `layers/base.py` | ✅ Done |
| F4 | Feature | **Major** | `layers/base.py` | ✅ Done |
| F5 | Feature | Minor | `layers/` (new) | ✅ Done |
| F6 | Feature | Minor | `layers/` (new) | ✅ Done |
| F7 | Feature | Minor | `utils/` (new) | ✅ Done |
| F8 | Feature | Minor | `layers/base.py` | ✅ Done |
| F9 | Feature | Minor | `__init__.py` | ✅ Done |
| F10 | Feature | Minor | `core/tensor.py` | ✅ Done |
| Q1 | Quality | **Major** | `pyproject.toml` | ✅ Done |
| Q2 | Quality | Minor | `test/session_test.py` | ✅ Done |
| Q3 | Quality | Minor | `examples/` | ✅ Done |
| Q4 | Quality | Minor | `ops/__init__.py` | ✅ Done |
| Q5 | Quality | Minor | `core/vjp_rules.py` | ✅ Done |
| Q6 | Quality | Minor | `core/vjp_rules.py` | ✅ Done |
| Q7 | Quality | Minor | `layers/` | ✅ Done |
| Q8 | Quality | Minor | `test/` (→ `scripts/`) | ✅ Done |
| T1–T10 | Testing | **Major** | `test/` | ✅ Done (T1–T10 all covered) |
| C1 | Compat | **Major** | `pyproject.toml` | ✅ Done |
| C2 | Compat | Minor | `core/backend.py` | ✅ Done |
| C3 | Compat | Minor | `core/backend.py` | ✅ Done |
| C4 | Compat | Minor | `core/vjp_rules.py` | ✅ Done (audited, clean) |

---

### Priority Order (recommended)

1. **Critical bugs** (B1–B3): Fix crashes in `RotaryPositionalEncoding`, `SinkhornTransport`, `NeuralODE`
2. **Security** (S1): Replace pickle with safe serialization
3. **Major bugs** (B4–B10): Fix wrong gradients, device switching, `__getitem__`
4. **Major performance** (P1–P2): Lazy grad allocation, cache RNN forward
5. **Major features** (F1, F2, F4): Gradient clipping, `requires_grad`, state dict
6. **Compatibility** (C1): Fix Python version in `pyproject.toml`
7. **Testing** (T1–T10): Add gradient correctness tests for all untested VJPs
8. Everything else (minor bugs, design, quality)
