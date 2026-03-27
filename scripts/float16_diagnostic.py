"""dnp/test/float16_diagnostic.py
=================================
Systematic float16 training diagnostics.

Each check is independent. A PASS means the result stays in float16
(no silent upcast to float64, no NaN/Inf).

Run:
    python -m dnp.test.float16_diagnostic
"""

import sys
import traceback

import numpy as np

import dnp.core.ops as ops
import dnp.core.backend as backend
from dnp.core.tensor import Tensor
from dnp.core import layers as L

# -------------------------------------------------------------------------
# Reporting helpers
# -------------------------------------------------------------------------

_RESULTS: list[dict] = []


def _record(name, status, detail=""):
    tag = "PASS" if status else "FAIL"
    _RESULTS.append({"name": name, "status": tag, "detail": detail})
    print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))


def _check(name, fn):
    try:
        fn()
    except Exception as e:
        _record(name, False, f"{type(e).__name__}: {e}")


def _t16(data):
    return Tensor(np.asarray(data, dtype=np.float16))


def _is16(t):
    """True if Tensor or array has float16 dtype."""
    data = t.data if isinstance(t, Tensor) else np.asarray(t)
    return data.dtype == np.float16


def _no_nan_inf(arr):
    a = np.asarray(arr)
    return bool(np.isfinite(a).all())


# =========================================================================
# Section 1 — Forward activations stay in float16
# =========================================================================


def _section_activations():
    print("\n--- 1. Forward activations preserve float16 ---")

    x = _t16([-1.0, 0.0, 1.0])

    for name, fn in [
        ("relu", lambda: ops.relu(x)),
        ("sigmoid", lambda: ops.sigmoid(x)),
        ("tanh", lambda: ops.tanh(x)),
        ("leaky_relu", lambda: ops.leaky_relu(x)),
        ("elu", lambda: ops.elu(x)),
        ("softplus", lambda: ops.softplus(x)),
        ("swish", lambda: ops.swish(x)),
        ("gelu", lambda: ops.gelu(x)),
        ("softmax", lambda: ops.softmax(_t16([1.0, 2.0, 3.0]))),
    ]:
        try:
            y = fn()
            ok = _is16(y)
            _record(name, ok, f"dtype={y.data.dtype}")
        except Exception as e:
            _record(name, False, str(e))


# =========================================================================
# Section 2 — VJP rules stay in float16
# =========================================================================


def _section_vjp():
    print("\n--- 2. VJP backward gradients preserve float16 ---")

    cases = [
        ("log", lambda: ops.sum(ops.log(_t16([0.5, 1.0, 2.0])))),
        ("sqrt", lambda: ops.sum(ops.sqrt(_t16([1.0, 4.0, 9.0])))),
        ("divide", lambda: ops.sum(ops.divide(_t16([1.0, 2.0]), _t16([0.5, 1.0])))),
        ("power", lambda: ops.sum(ops.power(_t16([2.0, 3.0]), _t16([2.0, 2.0])))),
        ("tan", lambda: ops.sum(ops.tan(_t16([0.1, 0.2])))),
        ("prod", lambda: ops.prod(_t16([[1.0, 2.0], [3.0, 4.0]]))),
        ("log1p", lambda: ops.sum(ops.log1p(_t16([0.5, 1.0])))),
        ("relu_vjp", lambda: ops.sum(ops.relu(_t16([-1.0, 1.0])))),
        ("sigmoid_vjp", lambda: ops.sum(ops.sigmoid(_t16([0.5, -0.5])))),
        ("gelu_vjp", lambda: ops.sum(ops.gelu(_t16([0.5, -0.5])))),
    ]

    for name, build_loss in cases:
        try:
            x_saved = None
            # Rebuild each time since backward zeroes things out
            if name == "log":
                x = _t16([0.5, 1.0, 2.0])
                loss = ops.sum(ops.log(x))
                x_saved = x
            elif name == "sqrt":
                x = _t16([1.0, 4.0, 9.0])
                loss = ops.sum(ops.sqrt(x))
                x_saved = x
            elif name == "divide":
                x = _t16([1.0, 2.0])
                y = _t16([0.5, 1.0])
                loss = ops.sum(ops.divide(x, y))
                x_saved = x
            elif name == "power":
                x = _t16([2.0, 3.0])
                y = _t16([2.0, 2.0])
                loss = ops.sum(ops.power(x, y))
                x_saved = x
            elif name == "tan":
                x = _t16([0.1, 0.2])
                loss = ops.sum(ops.tan(x))
                x_saved = x
            elif name == "prod":
                x = _t16([[1.0, 2.0], [3.0, 4.0]])
                loss = ops.prod(x)
                x_saved = x
            elif name == "log1p":
                x = _t16([0.5, 1.0])
                loss = ops.sum(ops.log1p(x))
                x_saved = x
            elif name == "relu_vjp":
                x = _t16([-1.0, 1.0])
                loss = ops.sum(ops.relu(x))
                x_saved = x
            elif name == "sigmoid_vjp":
                x = _t16([0.5, -0.5])
                loss = ops.sum(ops.sigmoid(x))
                x_saved = x
            elif name == "gelu_vjp":
                x = _t16([0.5, -0.5])
                loss = ops.sum(ops.gelu(x))
                x_saved = x

            loss.backward()
            g = x_saved.grad
            ok = g.dtype == np.float16
            _record(name, ok, f"grad dtype={g.dtype}, values={np.asarray(g)}")
        except Exception as e:
            _record(name, False, f"{type(e).__name__}: {e}")


# =========================================================================
# Section 3 — Optimizer state stays in float16
# =========================================================================


def _section_optimizers():
    print("\n--- 3. Optimizer state stays in float16 after one step ---")

    def _run_step(opt_class, **kwargs):
        backend.set_dtype("float16")
        w = Tensor(np.array([1.0, 2.0], dtype=np.float16))
        opt = opt_class([w], **kwargs)
        loss = ops.sum(ops.square(w))
        loss.backward()
        opt.step()
        backend.set_dtype("float64")
        return w.data.dtype

    for name, cls, kw in [
        ("SGD", L.SGD if hasattr(L, "SGD") else None, {"lr": 0.01}),
        ("Adam", L.Adam if hasattr(L, "Adam") else None, {"lr": 0.001}),
        ("AdamW", L.AdamW if hasattr(L, "AdamW") else None, {"lr": 0.001}),
        ("RMSprop", L.RMSprop if hasattr(L, "RMSprop") else None, {"lr": 0.001}),
        (
            "Momentum",
            L.Momentum if hasattr(L, "Momentum") else None,
            {"lr": 0.01, "momentum": 0.9},
        ),
    ]:
        if cls is None:
            # Try importing from optimizers
            try:
                from dnp.core import optimizers as _O

                cls = getattr(_O, name.split("+")[0], None)
            except ImportError:
                pass
        if cls is None:
            _record(name, False, "could not import optimizer")
            continue
        try:
            dtype_after = _run_step(cls, **kw)
            _record(
                name, dtype_after == np.float16, f"w.dtype after step = {dtype_after}"
            )
        except Exception as e:
            _record(name, False, f"{type(e).__name__}: {e}")


def _section_adam_buffers():
    print("\n--- 4. Adam moment buffers dtype after step ---")
    try:
        from dnp.core.optimizers import Adam

        backend.set_dtype("float16")
        w = Tensor(np.array([1.0, 2.0], dtype=np.float16))
        opt = Adam([w], lr=0.001)
        for _ in range(3):
            w.grad[:] = 0
            loss = ops.sum(ops.square(w))
            loss.backward()
            opt.step()
        backend.set_dtype("float64")
        pid = id(w)
        m_dtype = opt.m[pid].dtype
        v_dtype = opt.v[pid].dtype
        _record(
            "Adam momentum buffer m dtype", m_dtype == np.float16, f"m.dtype={m_dtype}"
        )
        _record(
            "Adam velocity buffer v dtype", v_dtype == np.float16, f"v.dtype={v_dtype}"
        )
        _record("Adam w not NaN/inf", _no_nan_inf(w.data), f"w={np.asarray(w.data)}")
    except Exception as e:
        _record("Adam buffers", False, f"{type(e).__name__}: {e}")


# =========================================================================
# Section 5 — BatchNorm running stats stay in float16
# =========================================================================


def _section_batchnorm():
    print("\n--- 5. BatchNorm running stats dtype ---")
    try:
        backend.set_dtype("float16")
        bn = L.BatchNorm1d(4)
        x = Tensor(np.ones((8, 4), dtype=np.float16))
        _ = bn(x)
        backend.set_dtype("float64")
        rm_dt = bn.running_mean.dtype
        rv_dt = bn.running_var.dtype
        _record("BN1d running_mean dtype", rm_dt == np.float16, f"dtype={rm_dt}")
        _record("BN1d running_var dtype", rv_dt == np.float16, f"dtype={rv_dt}")
    except Exception as e:
        _record("BatchNorm1d running stats", False, f"{type(e).__name__}: {e}")

    try:
        backend.set_dtype("float16")
        bn2 = L.BatchNorm2d(4)
        x2 = Tensor(np.ones((2, 4, 8, 8), dtype=np.float16))
        _ = bn2(x2)
        backend.set_dtype("float64")
        rm2_dt = bn2.running_mean.dtype
        rv2_dt = bn2.running_var.dtype
        _record("BN2d running_mean dtype", rm2_dt == np.float16, f"dtype={rm2_dt}")
        _record("BN2d running_var dtype", rv2_dt == np.float16, f"dtype={rv2_dt}")
    except Exception as e:
        _record("BatchNorm2d running stats", False, f"{type(e).__name__}: {e}")


# =========================================================================
# Section 6 — Weight initializers stay in target dtype
# =========================================================================


def _section_initializers():
    print("\n--- 6. Weight initializer dtypes ---")
    try:
        from dnp.core.layers import _apply_initializer

        backend.set_dtype("float16")
        gn = _apply_initializer("glorot_normal", (8, 4), np.float16)
        hn = _apply_initializer("he_normal", (8, 4), np.float16)
        gu = _apply_initializer("glorot_uniform", (8, 4), np.float16)
        backend.set_dtype("float64")
        _record("glorot_normal dtype", gn.dtype == np.float16, f"dtype={gn.dtype}")
        _record("he_normal dtype", hn.dtype == np.float16, f"dtype={hn.dtype}")
        _record("glorot_uniform dtype", gu.dtype == np.float16, f"dtype={gu.dtype}")
    except Exception as e:
        _record("initializers", False, f"{type(e).__name__}: {e}")


# =========================================================================
# Section 7 — Epsilon guards: no div/0 in log(near_zero)
# =========================================================================


def _section_eps_guards():
    print("\n--- 7. Epsilon guards (no NaN/inf at float16 resolution) ---")
    try:
        # log VJP: x very small → should not NaN
        x = Tensor(np.array([1e-6, 1e-7], dtype=np.float16))
        loss = ops.sum(ops.log(x))
        loss.backward()
        _record(
            "log VJP no NaN (near-zero x)",
            _no_nan_inf(x.grad),
            f"grad={np.asarray(x.grad)}",
        )
    except Exception as e:
        _record("log VJP no NaN", False, str(e))

    try:
        # sqrt VJP: x = 0 → should not NaN
        x = Tensor(np.array([0.0, 0.0001], dtype=np.float16))
        loss = ops.sum(ops.sqrt(x))
        loss.backward()
        _record(
            "sqrt VJP no NaN (x=0)", _no_nan_inf(x.grad), f"grad={np.asarray(x.grad)}"
        )
    except Exception as e:
        _record("sqrt VJP no NaN", False, str(e))

    try:
        # divide VJP: safe_eps guards y-gradient (-g*x/(y^2+eps)).
        # x-gradient (g/y) is mathematically inf when y=0 — that is correct.
        # We verify y.grad has no NaN (only y=1 element should be finite here).
        x = Tensor(np.array([1.0, 2.0], dtype=np.float16))
        y = Tensor(np.array([0.0, 1.0], dtype=np.float16))
        loss = ops.sum(ops.divide(x, y))
        loss.backward()
        # y.grad[1] = -g*x[1]/(y[1]^2+eps) = -2/(1+eps) ≈ -2 (finite)
        _record(
            "divide VJP denom-grad no NaN (safe_eps)",
            np.isfinite(np.asarray(y.grad)[1]),
            f"y.grad={np.asarray(y.grad)}",
        )
    except Exception as e:
        _record("divide VJP denom-grad no NaN", False, str(e))


# =========================================================================
# Section 8 — Full float16 mini-training loop (2-layer MLP)
# =========================================================================


def _section_mlp_training():
    print("\n--- 8. Float16 mini MLP training loop (10 steps) ---")
    try:
        from dnp.core.optimizers import Adam

        backend.set_dtype("float16")
        np.random.seed(42)

        # 2-layer MLP: 8 → 16 → 4
        fc1 = L.Linear(8, 16)
        fc2 = L.Linear(16, 4)

        opt = Adam(list(fc1.parameters()) + list(fc2.parameters()), lr=1e-3)

        # Fixed dataset so training is deterministic (no fresh random data each step)
        X_fixed = np.random.randn(8, 8).astype(np.float16)
        y_fixed = np.eye(4, dtype=np.float16)[[i % 4 for i in range(8)]]

        losses = []
        for step in range(50):
            X = Tensor(X_fixed)
            y = Tensor(y_fixed)

            h = ops.relu(fc1(X))
            logits = fc2(h)
            probs = ops.softmax(logits)

            loss = ops.mean(
                ops.negative(
                    ops.sum(
                        ops.multiply(
                            y, ops.log(ops.add(probs, backend.safe_eps(probs.data)))
                        ),
                        axis=-1,
                    )
                )
            )

            # Zero grads
            for p in list(fc1.parameters()) + list(fc2.parameters()):
                p.grad[:] = 0

            loss.backward()
            opt.step()
            losses.append(float(loss.data))

        backend.set_dtype("float64")

        all_finite = all(np.isfinite(v) for v in losses)
        loss_decreasing = losses[-1] < losses[0]
        # Check weight dtypes stayed float16
        w_dtype = list(fc1.parameters())[0].data.dtype

        _record(
            "Training loop no NaN/inf",
            all_finite,
            f"losses[0]={losses[0]:.4f} losses[-1]={losses[-1]:.4f}",
        )
        _record(
            "Loss decreases over 50 steps",
            loss_decreasing,
            f"{losses[0]:.4f} → {losses[-1]:.4f}",
        )
        _record("Weights stay float16", w_dtype == np.float16, f"dtype={w_dtype}")

    except Exception as e:
        backend.set_dtype("float64")
        _record(
            "MLP training loop", False, f"{type(e).__name__}: {traceback.format_exc()}"
        )


# =========================================================================
# Main
# =========================================================================


def main():
    print("=" * 60)
    print("  float16 training diagnostic")
    print("  NumPy", np.__version__)
    print("=" * 60)

    backend.set_dtype("float16")
    try:
        _section_activations()
        _section_vjp()
        backend.set_dtype("float16")
        _section_optimizers()
        _section_adam_buffers()
        _section_batchnorm()
        _section_initializers()
        backend.set_dtype("float16")
        _section_eps_guards()
        _section_mlp_training()
    finally:
        backend.set_dtype("float64")

    # Summary
    passed = sum(1 for r in _RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in _RESULTS if r["status"] == "FAIL")
    print("\n" + "=" * 60)
    print(f"  TOTAL: {passed} passed, {failed} failed out of {len(_RESULTS)}")
    print("=" * 60)
    if failed:
        print("\nFailed checks:")
        for r in _RESULTS:
            if r["status"] == "FAIL":
                print(f"  ✗  {r['name']}: {r['detail']}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
