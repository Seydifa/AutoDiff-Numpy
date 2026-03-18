"""
Example 1: Learning a Sobel Filter for Edge Detection
=======================================================

This example demonstrates learning a 2D convolution filter using automatic differentiation.
The model learns to detect edges in images by optimizing a filter to approximate the Sobel operator.

The computation graph tracks:
  - The input image
  - The learned filter parameters
  - The convolution operation
  - The loss computation
  - Backward passes through all operations
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
from scipy.signal import convolve2d


def sobel_filter_y():
    """Vertical edge detection filter (Sobel)."""
    return np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def conv2d_valid(x, w):
    """
    2D convolution with 'valid' mode (no padding).

    Args:
        x: Input image (H, W)
        w: Filter kernel (K, K)

    Returns:
        Output (H - K + 1, W - K + 1)
    """
    h, w_img = x.data.shape if hasattr(x, "data") else x.shape
    w_data = w.data if hasattr(w, "data") else w
    x_data = x.data if hasattr(x, "data") else x
    fh, fw = w_data.shape
    out_h = h - fh + 1
    out_w = w_img - fw + 1

    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(x_data[i : i + fh, j : j + fw] * w_data)
    return out


def main():
    print("=" * 70)
    print("Example 1: Learning a Convolution Filter for Edge Detection")
    print("=" * 70)

    # ========================================
    # 1. DATASET CREATION
    # ========================================
    print("\n1. Creating dataset...")

    # Create a simple image: white square on black background
    X_img = np.zeros((15, 15), dtype=np.float64)
    X_img[4:11, 4:11] = 1.0
    X_tensor = Tensor(X_img, name="InputImage")

    # Target: Sobel vertical edge detection
    W_sobel = sobel_filter_y()
    Y_target = convolve2d(X_img, W_sobel, mode="valid")
    Y_tensor = Tensor(Y_target, name="TargetEdges")

    print(f"  Input image shape: {X_img.shape}")
    print(f"  Target output shape: {Y_target.shape}")
    print(f"  True Sobel filter:\n{W_sobel}\n")

    # ========================================
    # 2. MODEL INITIALIZATION
    # ========================================
    print("2. Initializing learnable filter...")

    # Initialize filter with random values
    np.random.seed(42)
    W_learned = Tensor(np.random.randn(3, 3) * 0.1, name="LearnedFilter")
    print(f"  Initial filter shape: {W_learned.shape}")
    print(f"  Initial filter:\n{np.round(W_learned.data, 3)}\n")

    # ========================================
    # 3. TRAINING LOOP
    # ========================================
    print("3. Training (learning the filter)...")

    epochs = 300
    learning_rate = 0.05
    loss_history = []

    for epoch in range(epochs):
        # Reset computation graph to save memory
        session.reset()

        # --- Forward Pass ---
        # Convolution: pred = conv2d(X, W)
        # Use scipy convolve2d for the forward pass
        pred = dnp.ops.conv2d(X_tensor, W_learned, mode="valid")

        # MSE Loss: L = mean((pred - target)^2)
        diff = dnp.ops.subtract(pred, Y_tensor)
        diff_sq = dnp.ops.square(diff)
        loss = dnp.ops.mean(diff_sq)

        # --- Backward Pass ---
        W_learned.grad.fill(0.0)
        loss.backward()

        # --- Gradient Update (SGD) ---
        # Update the underlying data directly
        updated_data = np.asarray(W_learned) - learning_rate * W_learned.grad
        W_learned = Tensor(updated_data, name="LearnedFilter")

        loss_history.append(float(np.asarray(loss)))

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Loss: {loss.item():.6f}")

    session.reset()
    final_pred = convolve2d(X_img, W_learned.data, mode="valid")

    # Create figure with 5 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Learning a Edge Detection Filter (Sobel)", fontsize=16, fontweight="bold"
    )

    # Row 1: Input, Target, Prediction
    ax = axes[0, 0]
    ax.imshow(X_img, cmap="gray")
    ax.set_title("Input Image")
    ax.axis("off")

    ax = axes[0, 1]
    im = ax.imshow(Y_target, cmap="RdBu")
    ax.set_title("Target Edges (True Sobel)")
    ax.axis("off")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    im = ax.imshow(final_pred, cmap="RdBu")
    ax.set_title("Predicted Edges (Learned Filter)")
    ax.axis("off")
    plt.colorbar(im, ax=ax)

    # Row 2: Filters
    ax = axes[1, 0]
    im = ax.imshow(W_sobel, cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_title("True Sobel Filter")
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{W_sobel[i, j]:.1f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(W_learned.data, cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_title("Learned Filter")
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{W_learned.data[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    ax.plot(loss_history, color="#E63946", linewidth=2)
    ax.set_title("Training Loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if min(loss_history) > 0:
        ax.set_yscale("log")

    plt.tight_layout()

    # Save figure
    output_dir = (
        Path(__file__).parent.parent.parent / "figures" / "example_1_filter_learning"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "filter_learning_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")

    # Also create a loss curve figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_history, color="#E63946", linewidth=2.5, label="MSE Loss")
    ax.fill_between(range(len(loss_history)), loss_history, alpha=0.2, color="#E63946")
    ax.set_title("Filter Learning Convergence", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    if min(loss_history) > 0:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=12)

    plt.tight_layout()
    loss_path = output_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {loss_path}")

    plt.close("all")

    print("\n" + "=" * 70)
    print("Example 1 completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
