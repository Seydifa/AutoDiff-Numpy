"""
Example 4: Real-World Digit Recognition
=======================================

This example demonstrates training a Multi-Layer Perceptron (MLP) on a 
real-world dataset: the scikit-learn Digits dataset.
It showcases:
  - Multi-class classification (10 classes: 0 through 9)
  - Mini-batch training for better generalization
  - Softmax activation and Cross-Entropy loss computation
  - Real-world data preprocessing and train/test evaluation
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
from dnp.core.tensor import Tensor
from dnp.core.session import session
from dnp.layers import Module, Linear
from dnp.core.optimizers import Adam

class DigitsNetwork(Module):
    """
    MLP for 10-class digit classification.
    Input matches the 64 pixels (8x8) of the digits dataset.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(64, 64, name="Hidden1")
        self.fc2 = Linear(64, 32, name="Hidden2")
        self.fc3 = Linear(32, 10, name="Output")

    def forward(self, x):
        h = self.fc1(x)
        h = dnp.ops.relu(h)
        h = self.fc2(h)
        h = dnp.ops.relu(h)
        logits = self.fc3(h)
        return dnp.ops.softmax(logits)

def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot encoded vectors."""
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1.0
    return one_hot

def cross_entropy_loss(pred, target, epsilon=1e-8):
    """
    Calculate categorical cross-entropy loss.
    L = -mean(sum(target * log(pred + epsilon), axis=-1))
    """
    # Ensure numerical stability
    pred_safe = dnp.ops.add(pred, Tensor(np.array(epsilon)))
    log_pred = dnp.ops.log(pred_safe)
    
    # Element-wise multiply masked by targets (one-hot)
    loss_elems = dnp.ops.multiply(target, log_pred)
    
    # Sum over classes, mean over batch
    sum_classes = dnp.ops.sum(loss_elems, axis=-1)
    mean_batch = dnp.ops.mean(sum_classes)
    
    return dnp.ops.negative(mean_batch)

def evaluate_accuracy(model, X_np, y_np):
    """Evaluate accuracy of the model on the provided data."""
    session.reset()
    pred = model(Tensor(X_np, name="Eval_X"))
    pred_np = np.asarray(pred)
    predictions = np.argmax(pred_np, axis=-1)
    acc = np.mean(predictions == y_np)
    return acc, predictions

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
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-hot encode targets for training
    y_train_oh = one_hot_encode(y_train, 10)
    y_test_oh = one_hot_encode(y_test, 10)
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # ========================================
    # 2. MODEL INITIALIZATION
    # ========================================
    print("\n2. Initializing neural network...")
    
    np.random.seed(42)
    model = DigitsNetwork()
    
    print("  Architecture: 64 → [Dense(64, ReLU)] → [Dense(32, ReLU)] → Dense(10, Softmax)")
    
    param_count = sum(p.size for p in model.parameters())
    print(f"  Total parameters: {param_count}")

    # ========================================
    # 3. SETTINGS & CONSTANTS
    # ========================================
    print("\n3. Setting up optimizer and training configuration...")
    
    epochs = 40
    batch_size = 64
    optimizer = Adam(model.parameters(), lr=0.005)
    
    print(f"  Optimizer: Adam (lr=0.005)")
    print(f"  Batch size: {batch_size}")

    # ========================================
    # 4. TRAINING LOOP
    # ========================================
    print("\n4. Training the model...")
    
    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    
    train_loss_history = []
    test_acc_history = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle training data
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_oh[indices]
        
        for i in range(0, num_samples, batch_size):
            session.reset()
            optimizer.zero_grad()
            
            # Extract batch
            X_batch = Tensor(X_shuffled[i:i+batch_size], name="X_batch")
            Y_batch = Tensor(y_shuffled[i:i+batch_size], name="Y_batch")
            
            # Forward pass
            pred = model(X_batch)
            
            # Calculate loss
            loss = cross_entropy_loss(pred, Y_batch)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Record average epoch loss
        avg_loss = np.mean(epoch_losses)
        train_loss_history.append(avg_loss)
        
        if epoch == 0:
            out_path = Path(__file__).parent.parent.parent / "figures" / "example_4_digits_classification" / "computation_graph.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            session.show_graphe(title="Digits Classification Graph", save=True, filename=str(out_path))
            
        # Evaluate on test set
        test_acc, _ = evaluate_accuracy(model, X_test, y_test)
        test_acc_history.append(test_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:2d}/{epochs} | Train Loss: {avg_loss:.4f} | Test Acc: {test_acc*100:.1f}%")

    print("\n5. Training complete!")
    print(f"  Final Train Loss: {train_loss_history[-1]:.4f}")
    print(f"  Final Test Accuracy: {test_acc_history[-1]*100:.2f}%")

    # ========================================
    # 5. VISUALIZATIONS
    # ========================================
    print("\n6. Generating visualizations...")
    
    output_dir = Path(__file__).parent.parent.parent / "figures" / "example_4_digits_classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(train_loss_history, color="#E63946", linewidth=2.5)
    ax1.set_title("Training Loss (Cross-Entropy)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot([acc * 100 for acc in test_acc_history], color="#457B9D", linewidth=2.5)
    ax2.set_title("Test Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    curves_path = output_dir / "digits_training_curves.png"
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {curves_path}")
    
    # 2. Sample Predictions
    _, test_preds = evaluate_accuracy(model, X_test, y_test)
    
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    fig.suptitle("Sample Predictions on Test Set", fontsize=15, fontweight="bold", y=1.02)
    
    # Inverse transform to get visible images
    X_test_vis = scaler.inverse_transform(X_test).reshape(-1, 8, 8)
    
    for i, ax in enumerate(axes.flat):
        if i < len(X_test):
            ax.imshow(X_test_vis[i], cmap=plt.cm.gray_r, interpolation='nearest')
            pred_label = test_preds[i]
            true_label = y_test[i]
            
            color = "green" if pred_label == true_label else "red"
            ax.set_title(f"Pred: {pred_label} (True: {true_label})", color=color, fontsize=10)
            ax.axis('off')
            
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
