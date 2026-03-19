"""
Example 2: Learning the XOR Problem with Neural Networks
===========================================================

This example demonstrates learning a non-linear decision boundary.
The XOR problem is a classic example that cannot be solved by a linear model,
requiring at least one hidden layer with non-linear activations.

The neural network architecture:
  Input (2) → Hidden layer (8 neurons, ReLU) → Output (1, Sigmoid)

Key concepts demonstrated:
  - Module composition (building blocks)
  - Non-linear activation functions (ReLU, Sigmoid)
  - Adam optimizer for better convergence
  - Decision boundary visualization
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dnp
from dnp.core.tensor import Tensor
from dnp.core.session import session
from dnp.layers import Module, Linear
from dnp.core.optimizers import Adam


class XORNetwork(Module):
    """
    Neural network to solve the XOR problem.

    Architecture:
      Input (2) → Linear(8) → ReLU → Linear(1) → Sigmoid
    """

    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 8, name="Hidden")
        self.fc2 = Linear(8, 1, name="Output")

    def forward(self, x):
        """Forward pass through the network."""
        h = self.fc1(x)
        h = dnp.ops.relu(h)
        out = self.fc2(h)
        out = dnp.ops.sigmoid(out)
        return out


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

    X_tensor = Tensor(X_data, name="Input")
    Y_tensor = Tensor(Y_data, name="Target")

    print(f"  Data points:")
    for i in range(len(X_data)):
        print(f"    X{i} = {X_data[i]} → Y{i} = {Y_data[i, 0]:.0f}")

    # ========================================
    # 2. MODEL INITIALIZATION
    # ========================================
    print("\n2. Initializing neural network...")

    np.random.seed(42)
    model = XORNetwork()

    print("  Architecture: 2 → [Dense(8, ReLU)] → Dense(1, Sigmoid)")

    param_count = sum(p.size for p in model.parameters())
    print(f"  Total parameters: {param_count}")

    # ========================================
    # 3. OPTIMIZER SETUP
    # ========================================
    print("\n3. Setting up optimizer...")

    optimizer = Adam(model.parameters(), lr=0.05, beta1=0.9, beta2=0.999)
    print(f"  Optimizer: Adam (lr=0.05)")

    # ========================================
    # 4. TRAINING LOOP
    # ========================================
    print("\n4. Training the model...")

    epochs = 500
    loss_history = []

    for epoch in range(epochs):
        # Reset computation graph
        session.reset()
        optimizer.zero_grad()

        # --- Forward Pass ---
        pred = model(X_tensor)

        # MSE Loss
        diff = dnp.ops.subtract(pred, Y_tensor)
        loss = dnp.ops.mean(dnp.ops.square(diff))

        # --- Backward Pass ---
        loss.backward()

        # --- Optimizer Step ---
        optimizer.step()

        loss_history.append(float(np.asarray(loss)))

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Loss: {loss.item():.6f}")

        if epoch == 0:
            out_path = Path(__file__).parent.parent.parent / "figures" / "example_2_xor_problem" / "computation_graph.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            session.show_graphe(title="XOR Problem Graph", save=True, filename=str(out_path))

    print("\n5. Training complete!")
    print(f"  Final loss: {loss_history[-1]:.6f}\n")

    # ========================================
    # 5. EVALUATION
    # ========================================
    print("6. Evaluating on training data...")

    session.reset()
    final_pred = model(X_tensor)

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

    session.reset()
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
    sample_idx = 0
    session.reset()
    # Simulate predictions at different epochs (using final model)
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
