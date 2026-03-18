# DifferentialNumpy Examples

This directory contains comprehensive examples demonstrating the capabilities of the DifferentialNumpy (dnp) module for automatic differentiation and neural network training.

## Examples Overview

### Example 1: Learning a Convolution Filter for Edge Detection
**File:** `example_1_filter_learning/filter_learning.py`

This example demonstrates learning a 2D convolution filter using automatic differentiation. The model learns to detect edges in images by optimizing a filter to approximate the Sobel operator.

**Key Concepts:**
- Computation graph tracking
- Convolution operations
- MSE loss minimization
- Gradient descent optimization
- Filter visualization

**Generated Figures:**
- `figures/example_1_filter_learning/filter_learning_results.png` - Complete training results
- `figures/example_1_filter_learning/loss_curve.png` - Loss convergence curve

**To Run:**
```bash
python example_1_filter_learning/filter_learning.py
```

---

### Example 2: Learning the XOR Problem with Neural Networks
**File:** `example_2_xor_problem/xor_problem.py`

This example demonstrates learning a non-linear decision boundary using a multi-layer neural network. The XOR problem is a classic demonstration that linear models cannot solve - requiring non-linear activations.

**Network Architecture:**
- Input layer: 2 neurons (X1, X2)
- Hidden layer: 8 neurons with ReLU activation
- Output layer: 1 neuron with Sigmoid activation

**Key Concepts:**
- Module composition and layers
- Non-linear activation functions (ReLU, Sigmoid)
- Adam optimizer for enhanced convergence
- Decision boundary visualization
- Multi-sample training with batches

**Generated Figures:**
- `figures/example_2_xor_problem/xor_decision_boundary.png` - Decision boundary, loss, and probability maps
- `figures/example_2_xor_problem/training_dynamics.png` - Detailed training metrics

**To Run:**
```bash
python example_2_xor_problem/xor_problem.py
```

---

### Example 3: Simple Regression - Polynomial Fitting
**File:** `example_3_simple_regression/regression.py`

This example demonstrates function approximation using a neural network to fit a sine wave from noisy data. This showcases the network's ability to learn complex, non-linear functions.

**Network Architecture:**
- Input layer: 1 neuron (x)
- Hidden layer: 16 neurons with ReLU activation
- Output layer: 1 neuron (y)

**Dataset:**
- Training: 100 noisy samples from y = sin(x)
- Test: 200 clean samples for evaluation
- Function domain: [-2π, 2π]

**Key Concepts:**
- Synthetic data generation
- Train-test split (generalization analysis)
- Model capacity and regularization
- Residual analysis
- Multiple evaluation metrics (MSE, RMSE, MAE)

**Generated Figures:**
- `figures/example_3_simple_regression/regression_analysis.png` - 6-panel analysis
- `figures/example_3_simple_regression/regression_fit.png` - Fit visualization with confidence bands
- `figures/example_3_simple_regression/learning_curves.png` - Training and test curves

**To Run:**
```bash
python example_3_simple_regression/regression.py
```

### Example 4: Real-World Digit Recognition
**File:** `example_4_digits_classification/digits_classification.py`

This example demonstrates training a Multi-Layer Perceptron (MLP) on a real-world dataset: the scikit-learn Digits dataset.

**Network Architecture:**
- Input layer: 64 neurons (Flattened 8x8 image)
- Hidden Layer 1: 64 neurons with ReLU activation
- Hidden Layer 2: 32 neurons with ReLU activation
- Output layer: 10 neurons with Softmax activation

**Key Concepts:**
- Multi-class classification (10 classes)
- Mini-batch training for better generalization
- Softmax activation and custom Cross-Entropy loss computation
- Real-world data preprocessing (Standardization, one-hot encoding)

**Generated Figures:**
- `figures/example_4_digits_classification/digits_training_curves.png` - Loss and accuracy over epochs
- `figures/example_4_digits_classification/digits_sample_predictions.png` - Image samples with predictions

**To Run:**
```bash
python example_4_digits_classification/digits_classification.py
```

---

## Running All Examples

To run all examples and generate all figures:

```bash
python example_1_filter_learning/filter_learning.py
python example_2_xor_problem/xor_problem.py
python example_3_simple_regression/regression.py
python example_4_digits_classification/digits_classification.py
```

Or create a batch script (if needed):

```bash
#!/bin/bash
cd "$(dirname "$0")"
echo "Running all DifferentialNumpy examples..."
python example_1_filter_learning/filter_learning.py
python example_2_xor_problem/xor_problem.py
python example_3_simple_regression/regression.py
python example_4_digits_classification/digits_classification.py
echo "All examples completed!"
```

---

## Figure Locations

All generated figures are saved in organized directories:

```
figures/
├── example_1_filter_learning/
│   ├── filter_learning_results.png
│   └── loss_curve.png
├── example_2_xor_problem/
│   ├── xor_decision_boundary.png
│   └── training_dynamics.png
└── example_3_simple_regression/
    ├── regression_analysis.png
    ├── regression_fit.png
    └── learning_curves.png
```

---

## Key Features Demonstrated

| Feature | Example 1 | Example 2 | Example 3 |
|---------|-----------|-----------|-----------|
| Computation Graph | ✓ | ✓ | ✓ |
| Automatic Differentiation | ✓ | ✓ | ✓ |
| Linear Layers | ✓ | ✓ | ✓ |
| Conv2D Layers | ✓ | - | - |
| ReLU Activation | - | ✓ | ✓ |
| Sigmoid Activation | - | ✓ | - |
| SGD Optimizer | ✓ | - | ✓ |
| Adam Optimizer | - | ✓ | ✓ |
| Loss Minimization | ✓ | ✓ | ✓ |
| Visualization | ✓ | ✓ | ✓ |

---

## Dependencies

The examples require:
- `numpy` - Numerical computation
- `matplotlib` - Figure generation and visualization
- `dnp` - DifferentialNumpy module (included in parent directory)

Install dependencies:
```bash
pip install numpy matplotlib
```

---

## Educational Value

These examples are designed to:

1. **Illustrate Core Concepts:** Show how automatic differentiation works in practice
2. **Demonstrate Module System:** Show composition of layers into complete models
3. **Explore Optimization:** Display how different optimizers (SGD, Adam) converge
4. **Visualize Learning:** Display decision boundaries and function approximation
5. **Analyze Generalization:** Show training dynamics and overfitting detection

---

## Notes

- All random seeds are fixed for reproducibility
- Loss values are printed every 50-100 epochs for monitoring
- Computation graphs are reset between epochs to manage memory
- Figures are saved at high DPI (150) for publication quality

---

## Author Notes

Each example is self-contained and can be run independently. The code includes:
- Detailed docstrings explaining each section
- Inline comments for key operations
- Progress printing for user feedback
- Professional visualization with proper labels

For more information about the dnp module, see the main project documentation.
