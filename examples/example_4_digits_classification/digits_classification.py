"""
Example 4: Real-World Digit Recognition
=======================================

This example demonstrates training a Multi-Layer Perceptron (MLP) on the
scikit-learn Digits dataset.  It showcases the v3 features:

  - ``dnp.CrossEntropyLoss`` module (takes raw logits + integer labels)
  - ``dnp.Trainer`` with mini-batch training
  - ``dnp.ReduceLROnPlateau`` learning-rate scheduler
  - ``dnp.EarlyStopping`` callback to avoid over-fitting
  - ``Trainer.predict()`` under ``session.no_grad()`` for clean inference
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dnp
from dnp.core.session import session

Tensor = dnp.Tensor


class DigitsNetwork(dnp.Module):
    """MLP for 10-class digit classification (64 → 64 → 32 → 10 logits)."""

    def __init__(self):
        super().__init__()
        self.fc1 = dnp.Linear(64, 64, name="Hidden1")
        self.fc2 = dnp.Linear(64, 32, name="Hidden2")
        self.fc3 = dnp.Linear(32, 10, name="Output")

    def forward(self, x):
        h = dnp.ops.relu(self.fc1(x))
        h = dnp.ops.relu(self.fc2(h))
        return self.fc3(h)  # raw logits — CrossEntropyLoss applies softmax internally


def main():
    print("=" * 70)
    print("Example 4: Real-World Digit Recognition (Multi-class Classification)")
    print("=" * 70)

    # ========================================
    # 1. LOAD & PREPROCESS DATASET
    # ========================================
    print("\n1. Loading and preprocessing the digits dataset...")

    digits = load_digits()
    X = digits.data
    y = digits.target

    print(f"  Total samples: {X.shape[0]}")
    print(f"  Feature dimensions: {X.shape[1]} (8x8 images)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # CrossEntropyLoss expects integer labels — no one-hot needed
    y_train_f = y_train.reshape(-1, 1).astype(np.float64)
    y_test_f = y_test.reshape(-1, 1).astype(np.float64)

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # ========================================
    # 2. MODEL
    # ========================================
    print("\n2. Initializing neural network...")

    np.random.seed(42)
    model = DigitsNetwork()

    print("  Architecture: 64 → Dense(64,ReLU) → Dense(32,ReLU) → Dense(10) [logits]")
    print(f"  Total parameters: {sum(p.size for p in model.parameters())}")

    # ========================================
    # 3. LOSS, OPTIMIZER, SCHEDULER, CALLBACKS  (v3)
    # ========================================
    print("\n3. Setting up training components...")

    criterion = dnp.CrossEntropyLoss()  # accepts logits + integer labels
    optimizer = dnp.Adam(model.parameters(), lr=0.005)
    scheduler = dnp.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    early_stop = dnp.EarlyStopping(monitor="val_loss", patience=10, verbose=True)

    print(f"  Loss:      CrossEntropyLoss")
    print(f"  Optimizer: Adam (lr=0.005)")
    print(f"  Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
    print(f"  Callback:  EarlyStopping (patience=10, monitor=val_loss)")

    # Capture computation graph (one forward pass, no backward)
    out_path = (
        Path(__file__).parent.parent.parent
        / "figures"
        / "example_4_digits_classification"
        / "computation_graph.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.graph():
        _p = model(Tensor(X_train[:8], name="X"))
    session.show_graphe(
        title="Digits Classification Graph", save=True, filename=str(out_path)
    )

    # ========================================
    # 4. TRAINING — v3: Trainer.fit()
    # ========================================
    print("\n4. Training the model (Trainer.fit)...")

    trainer = dnp.Trainer(
        model,
        optimizer,
        loss_fn=lambda y_pred, y_true: criterion(y_pred, y_true),
        scheduler=scheduler,
        callbacks=[early_stop],
        verbose=True,
    )

    history = trainer.fit(
        X_train,
        y_train_f,
        epochs=40,
        batch_size=64,
        validation_data=(X_test, y_test_f),
    )

    train_loss_history = history.history.get("loss", [])
    val_loss_history = history.history.get("val_loss", [])

    # ========================================
    # 5. EVALUATION — v3: Trainer.predict()
    # ========================================
    print("\n5. Evaluating on test set...")

    pred_logits = trainer.predict(X_test)  # (N, 10) logits
    test_preds = np.argmax(pred_logits, axis=-1)
    test_acc = np.mean(test_preds == y_test)

    print(f"  Final Train Loss: {train_loss_history[-1]:.4f}")
    print(f"  Final Test Accuracy: {test_acc * 100:.2f}%")

    # ========================================
    # 6. VISUALIZATIONS
    # ========================================
    print("\n6. Generating visualizations...")

    output_dir = (
        Path(__file__).parent.parent.parent
        / "figures"
        / "example_4_digits_classification"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(train_loss_history, color="#E63946", linewidth=2.5, label="Train")
    if val_loss_history:
        axes[0].plot(val_loss_history, color="#457B9D", linewidth=2.5, label="Val")
        axes[0].legend()
    axes[0].set_title("Cross-Entropy Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Re-compute per-epoch accuracy from raw predictions for the plot
    axes[1].axhline(
        y=test_acc * 100,
        color="#457B9D",
        linewidth=2.5,
        linestyle="--",
        label=f"Final test acc: {test_acc * 100:.1f}%",
    )
    axes[1].set_title("Test Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 105)
    axes[1].legend()

    plt.tight_layout()
    curves_path = output_dir / "digits_training_curves.png"
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {curves_path}")

    # Sample Predictions
    fig, axes_grid = plt.subplots(3, 4, figsize=(10, 8))
    fig.suptitle(
        "Sample Predictions on Test Set", fontsize=15, fontweight="bold", y=1.02
    )

    X_test_vis = scaler.inverse_transform(X_test).reshape(-1, 8, 8)

    for i, ax in enumerate(axes_grid.flat):
        if i < len(X_test):
            ax.imshow(X_test_vis[i], cmap=plt.cm.gray_r, interpolation="nearest")
            pred_label = test_preds[i]
            true_label = y_test[i]
            color = "green" if pred_label == true_label else "red"
            ax.set_title(
                f"Pred: {pred_label} (True: {true_label})", color=color, fontsize=10
            )
            ax.axis("off")

    plt.tight_layout()
    samples_path = output_dir / "digits_sample_predictions.png"
    plt.savefig(samples_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {samples_path}")

    plt.close("all")

    print("\n" + "=" * 70)
    print("Example 4 completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
