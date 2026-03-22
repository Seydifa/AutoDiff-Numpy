"""dnp/test/benchmark.py
=====================
Runtime profiler and bottleneck analyser for dnp.

Measures three things per op:

  forward_only  — raw forward call (no graph recording)
  forward_graph — forward call through the Ops wrapper (Tensor creation, DAG)
  backward      — full reverse-mode pass from a scalar loss

``overhead = forward_graph - forward_only`` is the bookkeeping cost added by
the autograd engine per operation.

Usage
-----
    python -m dnp.test.benchmark             # all benchmarks + cProfile
    python -m dnp.test.benchmark --quick     # reduced iterations, no profiler

Output
------
  - Sorted timing table per category
  - cProfile top-20 hotspots (full run only)
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import timeit
from typing import Callable

import numpy as np

import dnp
import dnp.core.ops as ops
from dnp.core.tensor import Tensor
from dnp.core.session import session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensor(*shape, requires_scalar_loss: bool = True) -> Tensor:
    """Create a leaf Tensor with random float64 data."""
    return Tensor(np.random.randn(*shape).astype(np.float64))


def _reset():
    session.reset()


def _time_forward_only(fn: Callable, *args, reps: int = 200) -> float:
    """Time a raw NumPy/function call, bypassing the Ops wrapper."""
    t = timeit.timeit(lambda: fn(*args), number=reps)
    return t / reps * 1_000  # ms per call


def _time_forward_graph(op, *tensor_args, reps: int = 200) -> float:
    """Time one forward call through the Ops wrapper (Tensor in, Tensor out)."""

    def _call():
        _reset()
        return op(*tensor_args)

    t = timeit.timeit(_call, number=reps)
    return t / reps * 1_000


def _time_backward(op, *tensor_args, reps: int = 100) -> float:
    """Time one full forward+backward pass ending in a scalar loss."""

    def _pass():
        _reset()
        for t in tensor_args:
            if isinstance(t, Tensor):
                t.grad[:] = 0
        out_t = op(*tensor_args)
        out_t.backward()

    t = timeit.timeit(_pass, number=reps)
    return t / reps * 1_000


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------


def _build_suite(reps_fwd: int = 300, reps_bwd: int = 150) -> list[dict]:
    """Return a list of benchmark specs."""
    A = _make_tensor(64, 64)
    B = _make_tensor(64, 64)
    vec = _make_tensor(256)
    x3d = _make_tensor(8, 4, 16, 16)  # batch, ch, H, W
    w3d = _make_tensor(4, 4, 3, 3)  # out_ch, in_ch, kH, kW

    def _fwd(op, *args):
        raw = [a.data if isinstance(a, Tensor) else a for a in args]
        return _time_forward_only(op.func, *raw, reps=reps_fwd)

    def _graph(op, *args):
        return _time_forward_graph(op, *args, reps=reps_fwd)

    def _bwd(op, *args):
        return _time_backward(op, *args, reps=reps_bwd)

    suite = []

    def _reg(category, name, op, *args):
        suite.append(
            dict(
                category=category,
                name=name,
                fwd_only=_fwd(op, *args),
                fwd_graph=_graph(op, *args),
                bwd=_bwd(op, *args),
            )
        )

    # -- Binary --
    _reg("binary", "add", ops.add, A, B)
    _reg("binary", "subtract", ops.subtract, A, B)
    _reg("binary", "multiply", ops.multiply, A, B)
    _reg("binary", "divide", ops.divide, A, B)
    _reg("binary", "power", ops.power, A, B)
    _reg("binary", "matmul", ops.matmul, A, B)
    _reg("binary", "dot", ops.dot, A, B)
    _reg("binary", "maximum", ops.maximum, A, B)
    _reg("binary", "minimum", ops.minimum, A, B)

    # -- Unary --
    _reg("unary", "negative", ops.negative, A)
    _reg("unary", "square", ops.square, A)
    _reg("unary", "sqrt", ops.sqrt, A * A)  # positive input
    _reg("unary", "exp", ops.exp, A * 0.1)
    _reg("unary", "log", ops.log, ops.square(A) + 0.1)
    _reg("unary", "abs", ops.absolute, A)
    _reg("unary", "sign", ops.sign, A)

    # -- Trig --
    _reg("trig", "sin", ops.sin, vec)
    _reg("trig", "cos", ops.cos, vec)
    _reg("trig", "tanh", ops.tanh, vec)

    # -- Reductions --
    _reg("reduction", "sum(axis=0)", ops.sum, A)
    _reg("reduction", "mean(axis=0)", ops.mean, A)
    _reg("reduction", "max", ops.max, A)
    _reg("reduction", "min", ops.min, A)
    _reg("reduction", "prod", ops.prod, vec)

    # -- Shape --
    A_3d = _make_tensor(1, 64, 64)  # for squeeze
    _reg("shape", "reshape", ops.reshape, A, (4, 16 * 64))
    _reg("shape", "transpose", ops.transpose, A)
    _reg("shape", "expand_dims", ops.expand_dims, A, 0)
    _reg("shape", "squeeze", ops.squeeze, A_3d, 0)

    # -- Array manipulation --
    _reg("manip", "concatenate", ops.concatenate, A, B)
    _reg("manip", "stack", ops.stack, A, B)
    _reg("manip", "clip", ops.clip, A, -1.0, 1.0)
    _reg("manip", "cumsum", ops.cumsum, vec)
    _reg("manip", "flip", ops.flip, A)
    _reg("manip", "roll", ops.roll, A, 3)
    _reg("manip", "tile", ops.tile, vec, 2)
    _reg("manip", "repeat", ops.repeat, vec, 2)

    # -- Activations --
    _reg("activation", "sigmoid", ops.sigmoid, A)
    _reg("activation", "relu", ops.relu, A)
    _reg("activation", "leaky_relu", ops.leaky_relu, A)
    _reg("activation", "elu", ops.elu, A)
    _reg("activation", "softplus", ops.softplus, A)
    _reg("activation", "swish", ops.swish, A)
    _reg("activation", "gelu", ops.gelu, A)
    _reg("activation", "softmax", ops.softmax, A)

    # -- Pooling --
    _reg("pooling", "max_pool2d(2)", ops.max_pool2d, x3d, 2, 2, 0)
    _reg("pooling", "avg_pool2d(2)", ops.avg_pool2d, x3d, 2, 2, 0)

    return suite


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_COL = dict(CAT=14, NAME=22, FWD_O=12, FWD_G=12, OVERHEAD=12, BWD=12)
_TOTAL = sum(_COL.values()) + len(_COL) - 1


def _header():
    h = (
        f"{'Category':<{_COL['CAT']}}"
        f"{'Op':<{_COL['NAME']}}"
        f"{'fwd-only (ms)':>{_COL['FWD_O']}}"
        f"{'fwd+graph (ms)':>{_COL['FWD_G']}}"
        f"{'overhead (ms)':>{_COL['OVERHEAD']}}"
        f"{'backward (ms)':>{_COL['BWD']}}"
    )
    return h


def _row(r: dict) -> str:
    oh = r["fwd_graph"] - r["fwd_only"]
    return (
        f"{r['category']:<{_COL['CAT']}}"
        f"{r['name']:<{_COL['NAME']}}"
        f"{r['fwd_only']:>{_COL['FWD_O']}.4f}"
        f"{r['fwd_graph']:>{_COL['FWD_G']}.4f}"
        f"{oh:>{_COL['OVERHEAD']}.4f}"
        f"{r['bwd']:>{_COL['BWD']}.4f}"
    )


def _print_table(results: list[dict], title: str, sort_key: str = "bwd"):
    print(f"\n{'=' * _TOTAL}")
    print(f"  {title}")
    print("=" * _TOTAL)
    print(_header())
    print("-" * _TOTAL)
    for r in sorted(results, key=lambda x: x[sort_key], reverse=True):
        print(_row(r))
    print("=" * _TOTAL)


# ---------------------------------------------------------------------------
# cProfile helper
# ---------------------------------------------------------------------------


def _profile_end_to_end():
    """Run a small MLP forward+backward under cProfile and print top hotspots."""
    import dnp.core.layers as L

    np.random.seed(42)

    class _MLP(L.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = L.Linear(32, 64)
            self.fc2 = L.Linear(64, 32)
            self.fc3 = L.Linear(32, 10)

        def forward(self, x):
            x = ops.relu(self.fc1(x))
            x = ops.relu(self.fc2(x))
            return self.fc3(x)

    model = _MLP()
    X = Tensor(np.random.randn(16, 32))
    Y = Tensor(np.eye(10)[np.random.randint(0, 10, 16)])

    def _step():
        _reset()
        for p in model.parameters():
            p.grad[:] = 0
        out = model.forward(X)
        loss = ops.mean(ops.negative(ops.sum(Y * ops.log(ops.softmax(out) + 1e-8))))
        loss.backward()

    # warm-up
    for _ in range(5):
        _step()

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):
        _step()
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(20)
    print("\n" + "=" * _TOTAL)
    print("  cProfile — top 20 hotspots (50 MLP forward+backward steps)")
    print("=" * _TOTAL)
    # Print only the stats body, skip the long preamble
    lines = buf.getvalue().splitlines()
    for line in lines:
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="dnp runtime benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short run: reduced repetitions, no cProfile.",
    )
    parser.add_argument(
        "--sort",
        choices=["bwd", "fwd_only", "fwd_graph"],
        default="bwd",
        help="Column to sort the table by (default: backward).",
    )
    args = parser.parse_args(argv)

    reps_fwd = 50 if args.quick else 300
    reps_bwd = 30 if args.quick else 150

    print(f"\ndnp runtime benchmark  (fwd_reps={reps_fwd}, bwd_reps={reps_bwd})")
    print("Building benchmark suite …")

    results = _build_suite(reps_fwd=reps_fwd, reps_bwd=reps_bwd)
    _print_table(results, "All ops — sorted by backward time", sort_key=args.sort)

    # Per-category summary
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    print(f"\n{'=' * _TOTAL}")
    print("  Category totals (sum of backward times)")
    print("=" * _TOTAL)
    cat_totals = [
        (cat, sum(r["bwd"] for r in rows)) for cat, rows in categories.items()
    ]
    for cat, total in sorted(cat_totals, key=lambda x: x[1], reverse=True):
        print(f"  {cat:<20}  {total:.4f} ms")
    print("=" * _TOTAL)

    if not args.quick:
        _profile_end_to_end()


if __name__ == "__main__":
    main()
