# DifferentialNumpy (dnp)

**DifferentialNumpy** (`dnp`) is a lightweight, educational automatic differentiation engine and neural network library built entirely on top of NumPy (with SciPy for convolution). Inspired by PyTorch, it provides a dynamic computation graph, robust backpropagation via Vector-Jacobian Products (VJPs), and a full suite of neural network components — including a high-level `Trainer` API.
o
## 🚀 Key Features

* **Dynamic Computation Graph** — Tracks operations on `Tensor` objects and builds exact gradients using VJPs at every step.
* **`session.graph()` context manager** — Automatically scopes and resets the graph each forward pass, eliminating the memory leak caused by an ever-growing global session.
* **`session.no_grad()`** — Disables gradient tracking for clean inference or evaluation passes.
* **`graph_fn` decorator** — Wraps any callable so that raw NumPy/CuPy arrays and Python scalars are automatically promoted to `Tensor` before execution.
* **Versatile Layer Library (`dnp.layers`):**
  * Core: `Module`, `Sequential`, `Linear`, `Flatten`
  * Convolution & Pooling: `Conv2d`, `MaxPool2d`, `AvgPool2d`
  * Normalisation: `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`
  * Attention: `ScaledDotProductAttention`, `SelfAttention`, `MultiHeadAttention`
  * Regularisation: `Dropout`
  * Activations (as layers): `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
  * Loss modules: `CrossEntropyLoss`, `MSELoss`, `BCELoss`, `BCEWithLogitsLoss`
* **Optimizers** — `SGD`, `Momentum`, `RMSprop`, `Adagrad`, `Adam`, `AdamW` — all execute weight updates under `session.no_grad()` automatically.
* **LR Schedulers** — `StepLR`, `MultiStepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, `WarmupScheduler`.
* **`Trainer` + Callbacks** — High-level training loop with `EarlyStopping`, `ModelCheckpoint`, `ProgressLogger`, and a `History` object.
* **CUDA / GPU** — Transparent CuPy back-end; no code changes required.

## 📂 Project Structure

```text
DifferentialNumpy/
├── dnp/
│   ├── core/          # Tensor, SessionGraph, ops, layers, optimizers, VJP rules
│   ├── layers/        # Re-exports from core/layers.py (back-compat)
│   ├── ops/           # NumPy op wrappers, VJP_RULES.md
│   ├── utils/         # Trainer, EarlyStopping, ModelCheckpoint, ProgressLogger
│   └── test/          # pytest test suites
├── examples/          # End-to-end runnable Python scripts
├── notebooks/         # Jupyter demos (Auto_Diff, transformers, vision)
└── figures/           # PNG outputs from examples (graphs, loss curves, …)
```

## 🧠 Quick Start

Everything is accessible from the top-level `dnp` namespace:

```python
import numpy as np
import dnp
from dnp.core.session import session

# 1. Define model using Sequential + layer-activations
model = dnp.Sequential(
    dnp.Linear(10, 32),
    dnp.ReLU(),
    dnp.Linear(32, 2),
)

# 2. Loss, optimizer, scheduler
criterion = dnp.MSELoss()
optimizer = dnp.Adam(model.parameters(), lr=0.01)
scheduler = dnp.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# 3. High-level training  ──  Trainer handles the loop, graph resets & callbacks
trainer = dnp.Trainer(
    model, optimizer,
    loss_fn=lambda y_pred, y_true: criterion(y_pred, y_true),
    scheduler=scheduler,
    callbacks=[dnp.EarlyStopping(patience=20)],
)

X = np.random.randn(100, 10)
Y = np.random.randn(100, 2)
history = trainer.fit(X, Y, epochs=200, batch_size=32, validation_data=(X, Y))

# 4. Inference  ──  Trainer.predict() runs under session.no_grad()
preds = trainer.predict(X)   # numpy array (100, 2)
```

### Manual training loop (fine-grained control)

```python
import numpy as np
import dnp
from dnp.core.session import session

Tensor = dnp.Tensor

model = dnp.Sequential(dnp.Linear(10, 32), dnp.ReLU(), dnp.Linear(32, 1))
criterion = dnp.MSELoss()
optimizer = dnp.Adam(model.parameters(), lr=0.01)

X = np.random.randn(64, 10)
Y = np.random.randn(64, 1)

for epoch in range(100):
    # session.graph() resets the graph on entry — no memory leak
    with session.graph():
        pred = model(Tensor(X, name="x"))
        loss = criterion(pred, Tensor(Y, name="y"))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# Clean inference
with session.no_grad():
    out = model(Tensor(X, name="x"))
```

## 🛠️ Examples & Demonstrations

All examples live in `examples/` and use the v3 API. Run any of them from the project root:

```bash
python examples/example_2_xor_problem/xor_problem.py
```

---

### 1 — Learning a Convolution Filter (`example_1_filter_learning`)

Learns a 3×3 filter via backprop to replicate a Sobel edge-detector.  
Uses `session.graph()`, `dnp.MSELoss`, and `session.no_grad()` for evaluation.

| Final loss (300 epochs) |
|:-----------------------:|
| **0.0061** |

![Filter Learning Results](figures/example_1_filter_learning/filter_learning_results.png)
![Computation Graph](figures/example_1_filter_learning/computation_graph.png)

---

### 2 — The XOR Problem (`example_2_xor_problem`)

Solves XOR with `dnp.Sequential` + layer-level activations (`ReLU`, `Sigmoid`), `dnp.BCELoss`, and `dnp.Trainer` + `EarlyStopping`.

| Stopped at epoch | Final BCE loss | XOR predictions |
|:----------------:|:--------------:|:---------------:|
| **213 / 500** | **0.0016** | 0.0001, 1.0000, 0.9999, 0.0001 |

![XOR Decision Boundary](figures/example_2_xor_problem/xor_decision_boundary.png)
![Computation Graph](figures/example_2_xor_problem/computation_graph.png)

---

### 3 — Simple Regression (`example_3_simple_regression`)

Fits a `sin(x)` curve with a one-hidden-layer MLP using `dnp.Trainer` with validation data, `ReduceLROnPlateau` scheduler, and `EarlyStopping`.

| Stopped at epoch | Test MSE | Test RMSE | Test MAE |
|:----------------:|:--------:|:---------:|:--------:|
| **58 / 300** | **0.316** | **0.562** | **0.491** |

![Regression Fit](figures/example_3_simple_regression/regression_fit.png)
![Computation Graph](figures/example_3_simple_regression/computation_graph.png)

---

### 4 — Digit Recognition (`example_4_digits_classification`)

Trains an MLP on the scikit-learn Digits dataset using `dnp.CrossEntropyLoss` (raw logits, no manual one-hot), `dnp.Trainer`, `ReduceLROnPlateau`, and `EarlyStopping`.

| Stopped at epoch | Test Accuracy |
|:----------------:|:-------------:|
| **11 / 40** | **80.3 %** |

![Digit Predictions](figures/example_4_digits_classification/digits_sample_predictions.png)
![Computation Graph](figures/example_4_digits_classification/computation_graph.png)

---

## ⚙️ Installation & Requirements

Ensure Python 3.8+. The project uses **Poetry** for reproducible environments.

```bash
# Using Poetry
curl -sSL https://install.python-poetry.org | python3 -
poetry install
poetry shell

# Or plain pip
pip install numpy scipy matplotlib pytest networkx scikit-learn
```

Run the test suite:
```bash
pytest dnp/test/
```

## ⚡ CUDA / GPU Acceleration

DifferentialNumpy transparently switches to [CuPy](https://cupy.dev/) when it is installed — no code changes needed.

```bash
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda11x   # CUDA 11.x
```

```python
from dnp.core.tensor import Tensor

x = Tensor(np.random.randn(32, 64), device="cuda")  # direct GPU
x = x.cuda()   # move existing tensor to GPU
x = x.cpu()    # back to CPU
```

### Global dtype control

```python
import dnp

dnp.set_dtype('float32')   # halves memory; good for GPU
dnp.set_dtype('float64')   # default (highest precision)
print(dnp.get_dtype())
```

### Backend utilities

```python
from dnp.core import backend

backend.is_cuda_available          # True if CuPy is importable
backend.get_device_count()         # number of CUDA GPUs
backend.synchronize()              # block until GPU kernels finish
backend.as_numpy(array)            # GPU → CPU numpy
backend.as_cupy(array)             # CPU → GPU
backend.to_device(array, "cuda")   # generic transfer
```

## 🎓 Understanding Automatic Differentiation

The engine propagates gradients using **Vector-Jacobian Products (VJPs)** defined for every operator.

> [!TIP]
> Read [dnp/ops/VJP_RULES.md](dnp/ops/VJP_RULES.md) for the full mathematical derivation of each VJP rule.

Each example also saves a `computation_graph.png` — a visualisation of the forward DAG built by `session` — so you can literally see the graph your model constructs.

