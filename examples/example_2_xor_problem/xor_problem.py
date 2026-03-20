"""
Example 2: Learning the XOR Problem with Neural Networks
===========================================================

This example demonstrates learning a non-linear decision boundary.
It showcases the v3 features:

  - ``dnp.Sequential`` + layer-based activations (``dnp.ReLU``, ``dnp.Sigmoid``)
  - ``dnp.BCELoss`` loss module
  - ``dnp.Trainer`` high-level training loop with callbacks
  - ``dnp.EarlyStopping`` to avoid over-training
  - ``session.no_grad()`` for side-effect-free inference
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dnp
from dnp.core.session import session

# Use top-level exports (v3)
Tensor = dnp.Tensor


def main():
    print("=" * 70)
    print("Example 2: Learning the XOR Problem with Neural Networks")
    print("=" * 70)

    # ========================================
    # 1. DATASET CREATION
    # ========================================
    print("\n1. Creating XOR dataset...")

    X_data = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64
    )
    Y_data = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)

    print(f"  Data points:")
    for i in range(len(X_data)):
        print(f"    X{i} = {X_data[i]} → Y{i} = {Y_data[i, 0]:.0f}")

    # ========================================
    # 2. MODEL — v3: dnp.Sequential with layer activations
    # ========================================
    print("\n2. Initializing neural network (Sequential)...")

    np.random.seed(42)
    model = dnp.Sequential(
        dnp.Linear(2, 8, name="Hidden"),
        dnp.ReLU(),
        dnp.Linear(8, 1, name="Output"),
        dnp.Sigmoid(),
    )

    print("  Architecture: 2 → Linear(8) → ReLU → Linear(1) → Sigmoid")
    param_count = sum(p.size for p in model.parameters())
    print(f"  Total parameters: {param_count}")

    # ========================================
    # 3. LOSS, OPTIMIZER, CALLBACKS  (v3)
    # ========================================
    print("\n3. Setting up training components...")

    criterion = dnp.BCELoss()
    optimizer = dnp.Adam(model.parameters(), lr=0.05)
    early_stop = dnp.EarlyStopping(monitor="loss", patience=50, verbose=True)

    print(f"  Loss:      BCELoss")
    print(f"  Optimizer: Adam (lr=0.05)")
    print(f"  Callback:  EarlyStopping (patience=50)")

    # Capture computation graph before training starts
    out_path = (
        Path(__file__).parent.parent.parent
        / "figures"
        / "example_2_xor_problem"
        / "computation_graph.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.graph():
        _p = model(Tensor(X_data, name="Input"))
    session.show_graphe(title="XOR Problem Graph", save=True, filename=str(out_path))

    # ========================================
    # 4. TRAINING — v3: Trainer.fit()
    # ========================================
    print("\n4. Training the model (Trainer.fit)...")

    trainer = dnp.Trainer(
        model,
        optimizer,
        loss_fn=lambda y_pred, y_true: criterion(y_pred, y_true),
        callbacks=[early_stop],
        verbose=True,
    )

    history = trainer.fit(X_data, Y_data, epochs=500, batch_size=4, shuffle=False)
    loss_history = history.history["loss"]

    print(f"\n5. Training complete!  Final loss: {loss_history[-1]:.6f}\n")

    # ========================================
    # 5. EVALUATION — v3: session.no_grad()
    # ========================================
    print("6. Evaluating on training data...")

    with session.no_grad():
        final_pred = model(Tensor(X_data, name="Input"))

    print("  Predictions (should be close to [0, 1, 1, 0]):")
    final_pred_data = np.asarray(final_pred).flatten()
    for i in range(len(X_data)):
        pred_val = float(final_pred_data[i])
        target_val = float(Y_data[i, 0])
        print(f"    X{i} → Pred: {pred_val:.4f}, Target: {target_val:.0f}")

    # ========================================
    # 6. VISUALIZATION
    # ========================================
    print("\n7. Generating visualizations...")

    # Create a mesh for decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    with session.no_grad():
        grid_pred = model(Tensor(grid, name="Grid"))
    Z = np.asarray(grid_pred).reshape(xx.shape)

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 6))

    # --- Subplot 1: Loss Curve ---
    ax = plt.subplot(1, 3, 1)
    ax.plot(loss_history, color="#E63946", linewidth=2.5)
    ax.fill_between(range(len(loss_history)), loss_history, alpha=0.2, color="#E63946")
    ax.set_title("Training Loss (MSE)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_yscale("log")

    # --- Subplot 2: Decision Boundary ---
    ax = plt.subplot(1, 3, 2)
    contour = ax.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.7)
    contour_lines = ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

    # Overlay training points
    colors = Y_data.flatten()
    scatter = ax.scatter(
        X_data[:, 0],
        X_data[:, 1],
        c=colors,
        s=200,
        cmap="RdBu",
        edgecolors="black",
        linewidths=2,
        vmin=0,
        vmax=1,
        zorder=5,
    )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title("Decision Boundary", fontsize=13, fontweight="bold")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: Probability Heat Map ---
    ax = plt.subplot(1, 3, 3)
    im = ax.contourf(xx, yy, Z, levels=20, cmap="hot")
    ax.scatter(
        X_data[:, 0],
        X_data[:, 1],
        c=colors,
        s=200,
        cmap="RdBu",
        edgecolors="white",
        linewidths=2,
        vmin=0,
        vmax=1,
        zorder=5,
    )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title("Output Probability", fontsize=13, fontweight="bold")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(Class=1)")

    plt.suptitle(
        "XOR Problem: Non-linear Decision Boundary",
        fontsize=15,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()

    # Save figure
    output_dir = (
        Path(__file__).parent.parent.parent / "figures" / "example_2_xor_problem"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "xor_decision_boundary.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")

    # Create additional training dynamics figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss on log scale
    ax = axes[0, 0]
    ax.semilogy(loss_history, color="#E63946", linewidth=2)
    ax.set_title("Loss (Log Scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Loss on linear scale (last 100 epochs)
    ax = axes[0, 1]
    ax.plot(loss_history[-100:], color="#457B9D", linewidth=2)
    ax.set_title("Loss (Last 100 Epochs)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Predictions scatter
    ax = axes[1, 0]
    predictions = np.asarray(final_pred).flatten()
    targets = Y_data.flatten()
    ax.scatter(
        targets,
        predictions,
        s=150,
        alpha=0.7,
        color="#F1FAEE",
        edgecolors="#E63946",
        linewidths=2,
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect prediction")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title("Predictions vs Targets")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

    # Training dynamics
    ax = axes[1, 1]
    # Display final predictions per sample
    ax.barh(
        ["X[0,0]→0", "X[0,1]→1", "X[1,0]→1", "X[1,1]→0"],
        predictions.tolist(),
        color=["#E63946", "#457B9D", "#457B9D", "#E63946"],
    )
    ax.set_xlim(0, 1)
    ax.set_title("Final Predictions for Each Sample")
    ax.set_xlabel("Prediction Value")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    dynamics_path = output_dir / "training_dynamics.png"
    plt.savefig(dynamics_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {dynamics_path}")

    plt.close("all")

    print("\n" + "=" * 70)
    print("Example 2 completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
