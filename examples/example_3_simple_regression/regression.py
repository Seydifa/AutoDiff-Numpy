"""
Example 3: Simple Regression with Polynomial Fitting
======================================================

This example demonstrates learning a polynomial function from data.
We use automatic differentiation to fit a neural network to a non-linear function.

This showcase:
  - Simple dataset generation (sine wave + noise)
  - Single hidden layer network
  - SGD optimizer
  - Multiple loss metric visualizations
  - Model capacity analysis
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
from dnp.core.optimizers import SGD, Adam


class RegressionNetwork(Module):
    """
    Simple regression network with one hidden layer.

    Architecture:
      Input (1) → Linear(16) → ReLU → Linear(1)
    """

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc1 = Linear(1, hidden_dim, name="Hidden")
        self.fc2 = Linear(hidden_dim, 1, name="Output")

    def forward(self, x):
        """Forward pass through the network."""
        h = self.fc1(x)
        h = dnp.ops.relu(h)
        out = self.fc2(h)
        return out


def generate_synthetic_data(n_samples=100, noise_std=0.1):
    """
    Generate synthetic dataset: y = sin(x) + noise

    Args:
        n_samples: Number of training samples
        noise_std: Standard deviation of Gaussian noise

    Returns:
        X: Input features (n_samples, 1)
        Y: Target values (n_samples, 1)
    """
    np.random.seed(42)

    X = np.linspace(-2.0 * np.pi, 2.0 * np.pi, n_samples, dtype=np.float64).reshape(
        -1, 1
    )
    Y = np.sin(X) + np.random.normal(0, noise_std, size=(n_samples, 1))

    return X, Y


def main():
    print("=" * 70)
    print("Example 3: Simple Regression - Polynomial Fitting")
    print("=" * 70)

    # ========================================
    # 1. DATASET CREATION
    # ========================================
    print("\n1. Generating synthetic dataset (sin wave)...")

    X_train, Y_train = generate_synthetic_data(n_samples=100, noise_std=0.15)

    # Test data (without noise, for clean evaluation)
    X_test = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 200, dtype=np.float64).reshape(
        -1, 1
    )
    Y_test = np.sin(X_test)

    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Input range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  Target range: [{Y_train.min():.2f}, {Y_train.max():.2f}]")

    X_train_tensor = Tensor(X_train, name="X_Train")
    Y_train_tensor = Tensor(Y_train, name="Y_Train")

    X_test_tensor = Tensor(X_test, name="X_Test")
    Y_test_tensor = Tensor(Y_test, name="Y_Test")

    # ========================================
    # 2. MODEL INITIALIZATION
    # ========================================
    print("\n2. Initializing regression network...")

    np.random.seed(42)
    model = RegressionNetwork(hidden_dim=16)

    print("  Architecture: 1 → Dense(16, ReLU) → Dense(1)")

    param_count = sum(p.size for p in model.parameters())
    print(f"  Total parameters: {param_count}")

    # ========================================
    # 3. OPTIMIZER SETUP
    # ========================================
    print("\n3. Setting up optimizer...")

    optimizer = Adam(model.parameters(), lr=0.01)
    print(f"  Optimizer: Adam (lr=0.01)")

    # ========================================
    # 4. TRAINING LOOP
    # ========================================
    print("\n4. Training the model...")

    epochs = 300
    train_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):
        # --- Training Phase ---
        session.reset()
        optimizer.zero_grad()

        # Forward pass
        train_pred = model(X_train_tensor)

        # MSE Loss
        train_diff = dnp.ops.subtract(train_pred, Y_train_tensor)
        train_loss = dnp.ops.mean(dnp.ops.square(train_diff))

        # Backward pass
        train_loss.backward()

        # Optimizer step
        optimizer.step()

        train_loss_history.append(float(np.asarray(train_loss)))

        # --- Test Phase (no gradient) ---
        session.reset()
        test_pred = model(X_test_tensor)
        test_diff = dnp.ops.subtract(test_pred, Y_test_tensor)
        test_loss = dnp.ops.mean(dnp.ops.square(test_diff))
        test_loss_history.append(float(np.asarray(test_loss)))

        if (epoch + 1) % 60 == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{epochs} | Train Loss: {train_loss.item():.6f} | "
                f"Test Loss: {test_loss.item():.6f}"
            )

    print("\n5. Training complete!")
    print(f"  Final train loss: {train_loss_history[-1]:.6f}")
    print(f"  Final test loss: {test_loss_history[-1]:.6f}")

    # ========================================
    # 5. FINAL EVALUATION
    # ========================================
    print("\n6. Evaluating on test set...")

    session.reset()
    final_pred = model(X_test_tensor)
    final_pred_data = np.asarray(final_pred).flatten()

    # Compute metrics
    mse = np.mean((final_pred_data - Y_test.flatten()) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(final_pred_data - Y_test.flatten()))

    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")

    # ========================================
    # 6. VISUALIZATION
    # ========================================
    print("\n7. Generating visualizations...")

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))

    # --- Subplot 1: Regression Fit ---
    ax = plt.subplot(2, 3, 1)
    ax.scatter(
        X_train, Y_train, alpha=0.6, s=50, label="Training data", color="#E63946"
    )
    ax.plot(X_test, Y_test, "k-", linewidth=2, label="True function", alpha=0.7)
    ax.plot(
        X_test, final_pred_data, "b-", linewidth=2, label="Learned model", alpha=0.8
    )
    ax.fill_between(
        X_test.flatten(),
        final_pred_data,
        Y_test.flatten(),
        alpha=0.2,
        color="#457B9D",
        label="Prediction error",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Regression Fit: Train vs Test", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Residuals ---
    ax = plt.subplot(2, 3, 2)
    residuals = final_pred_data - Y_test.flatten()
    ax.scatter(X_test, residuals, alpha=0.7, s=50, color="#E63946")
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.fill_between(X_test.flatten(), residuals, 0, alpha=0.2, color="#E63946")
    ax.set_xlabel("x")
    ax.set_ylabel("Residuals (Pred - True)")
    ax.set_title("Prediction Residuals", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: Prediction vs Target ---
    ax = plt.subplot(2, 3, 3)
    ax.scatter(Y_test.flatten(), final_pred_data, alpha=0.7, s=50, color="#457B9D")
    min_val = min(Y_test.min(), final_pred_data.min())
    max_val = max(Y_test.max(), final_pred_data.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect fit"
    )
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title("Predictions vs Ground Truth", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # --- Subplot 4: Training Loss ---
    ax = plt.subplot(2, 3, 4)
    ax.plot(train_loss_history, label="Train", linewidth=2, color="#E63946")
    ax.plot(test_loss_history, label="Test", linewidth=2, color="#457B9D")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training Progress", fontweight="bold")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Subplot 5: Loss Difference ---
    ax = plt.subplot(2, 3, 5)
    loss_diff = np.array(test_loss_history) - np.array(train_loss_history)
    ax.plot(loss_diff, linewidth=2, color="#A23B72")
    ax.fill_between(range(len(loss_diff)), loss_diff, alpha=0.3, color="#A23B72")
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss - Train Loss")
    ax.set_title("Generalization Gap", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # --- Subplot 6: Error Histogram ---
    ax = plt.subplot(2, 3, 6)
    ax.hist(residuals, bins=30, alpha=0.7, color="#457B9D", edgecolor="black")
    ax.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Zero error")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution", fontweight="bold")
    ax.legend()

    plt.suptitle(
        "Polynomial Regression: Function Approximation with Neural Network",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    # Save figure
    output_dir = (
        Path(__file__).parent.parent.parent / "figures" / "example_3_simple_regression"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "regression_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")

    # Create additional focused fit plot
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(
        X_train,
        Y_train,
        alpha=0.5,
        s=60,
        label="Training samples",
        color="#E63946",
        edgecolors="darkred",
        linewidths=0.5,
    )
    ax.plot(X_test, Y_test, "k-", linewidth=3, label="True function: sin(x)", alpha=0.8)
    ax.plot(
        X_test,
        final_pred_data,
        "#457B9D",
        linewidth=2.5,
        label="Neural network prediction",
        linestyle="--",
        alpha=0.9,
    )

    # Confidence band (based on residual std)
    residual_std = np.std(residuals)
    ax.fill_between(
        X_test.flatten(),
        final_pred_data - 2 * residual_std,
        final_pred_data + 2 * residual_std,
        alpha=0.2,
        color="#457B9D",
        label="±2σ confidence band",
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        "Non-linear Regression: Learning sin(x) from Noisy Data",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    fit_path = output_dir / "regression_fit.png"
    plt.savefig(fit_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {fit_path}")

    # Create learning curve comparison
    fig, ax = plt.subplots(figsize=(12, 7))

    epochs_array = np.arange(len(train_loss_history))

    ax.semilogy(
        epochs_array,
        train_loss_history,
        linewidth=2.5,
        label="Training loss",
        color="#E63946",
        marker="o",
        markevery=30,
        markersize=6,
    )
    ax.semilogy(
        epochs_array,
        test_loss_history,
        linewidth=2.5,
        label="Test loss",
        color="#457B9D",
        marker="s",
        markevery=30,
        markersize=6,
    )

    ax.fill_between(
        epochs_array,
        train_loss_history,
        test_loss_history,
        alpha=0.2,
        color="#A23B72",
        label="Generalization gap",
    )

    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE, log scale)", fontsize=12)
    ax.set_title(
        "Learning Curves: Convergence and Generalization",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")

    plt.tight_layout()
    learning_curve_path = output_dir / "learning_curves.png"
    plt.savefig(learning_curve_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {learning_curve_path}")

    plt.close("all")

    print("\n" + "=" * 70)
    print("Example 3 completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
