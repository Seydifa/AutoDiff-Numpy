# A Beginner's Guide to Vector-Jacobian Products (VJPs) in DifferentialNumpy

Welcome! If you are looking at the source code for `dnp/core/vjp_rules.py` and wondering what all these `lambda g, x, y: ...` functions are doing, you are in the right place. 

This document provides a gentle, accessible, and intuitive explanation of Vector-Jacobian Products (VJPs) and how reverse-mode Automatic Differentiation (Autograd) works under the hood.

---

## 1. The Big Picture: Why do we need VJPs?

In deep learning and machine learning, we train models by minimizing a **Loss** (a single number, or scalar). To minimize the loss, we need to know how every parameter in our model contributed to it. We need the **gradient** of the Loss with respect to every weight and input.

### The Problem with Jacobians
Imagine a simple operation in our neural network: an intermediate layer takes a vector $x$ (with 1,000 elements) and produces a vector $y$ (with 1,000 elements). 

The matrix that describes how every element of $y$ changes with respect to every element of $x$ is called the **Jacobian Matrix**. In this case, the Jacobian is a $1,000 \times 1,000$ matrix (1 million elements!).
If we had to compute and store this massive matrix for every single mathematical operation in our network, we would run out of memory instantly.

### The VJP Solution
Here is the trick: **we never actually need the full Jacobian matrix!** 
Because our final output (the Loss) is just a single number, the chain rule of calculus tells us we can compute gradients working *backwards* from the Loss.

During the backward pass (backpropagation), an operation $y = f(x)$ receives the gradient of the Loss with respect to its output $y$. Let's call this incoming gradient the **Vector** $g$.
$$ g = \frac{\partial Loss}{\partial y} $$

Our goal is to figure out the gradient of the Loss with respect to the input $x$:
$$ \frac{\partial Loss}{\partial x} = \frac{\partial Loss}{\partial y} \cdot \frac{\partial y}{\partial x} $$
$$ \frac{\partial Loss}{\partial x} = g \cdot J $$

Instead of building $J$ and multiplying it by $g$, we just write a specific rule for $g \cdot J$ directly. This is called the **Vector-Jacobian Product (VJP)**.

---

## 2. A Simple Example: Multiplication

Let's look at a concrete example. Suppose we have a mathematical operation: 
$$ z = x \cdot y $$

1. **Forward Pass:** We compute $z = x \cdot y$ and save $x$ and $y$ because we will need them later.
2. **Backward Pass:** Later on, the network gives us $g$, which is $\frac{\partial Loss}{\partial z}$.
3. **Calculus Rule:** 
   - How does $z$ change if we change $x$? By $y$. (So, $\frac{\partial z}{\partial x} = y$)
   - How does $z$ change if we change $y$? By $x$. (So, $\frac{\partial z}{\partial y} = x$)
4. **The VJP:**
   - The gradient for $x$ is the incoming gradient $g$ multiplied by the local derivative: `grad_x = g * y`
   - The gradient for $y$ is the incoming gradient $g$ multiplied by the local derivative: `grad_y = g * x`

In `DifferentialNumpy`, this rule is written exactly like this:
```python
np.multiply: lambda g, x, y: (g * y, g * x)
```
*Note: The actual code includes an `unbroadcast` function to handle dimensions, which we'll explain later!*

---

## 3. How VJPs are Structured in DifferentialNumpy

In `dnp/core/vjp_rules.py`, you will see a dictionary named `VJP_RULES`. It maps NumPy operations to their backward VJP functions. 

The format is always:
```python
operation: lambda g, input1, input2, ... : (gradient_for_input1, gradient_for_input2, ...)
```
Where `g` is always the incoming gradient from the layer above it.

Here is a breakdown of the rules in plain English:

### A. Unary Operations (One Input)
These operations take one input $x$ and produce one output.
* **Negative (`np.negative`):** $y = -x$. A change in $x$ causes the opposite change in $y$.
  * Rule: `lambda g, x: (-g,)`
* **Logarithm (`np.log`):** $y = \ln(x)$. The derivative of $\ln(x)$ is $1/x$. 
  * Rule: `lambda g, x: (g / (x + EPSILON),)` (We add a tiny EPSILON to prevent dividing by zero).
* **Exponential (`np.exp`):** $y = e^x$. The derivative of $e^x$ is $e^x$.
  * Rule: `lambda g, x: (g * np.exp(x),)`

### B. Matrix Multiplication (`np.matmul`, `np.dot`)
Why isn't matrix multiplication just `g * y`? Because matrix shapes matter! 
If $Z = X @ Y$, then:
* To find the gradient for $X$, we multiply the incoming gradient $g$ by $Y$ transposed ($Y^T$).
* To find the gradient for $Y$, we multiply $X$ transposed ($X^T$) by the incoming gradient $g$.
* Rule: `lambda g, x, y: (np.matmul(g, y.T), np.matmul(x.T, g))`

### C. Reductions (Sum, Mean, Max)
When you reduce an array (like taking the sum of a vector), you are collapsing numbers into a single number. In the backward pass, we have to "un-collapse" (expand) the gradient back to the original shape.
* **Sum (`np.sum`):** If $y = x_1 + x_2 + x_3$, then changing any $x_i$ by 1 changes the sum by 1. Therefore, every element gets the *exact same* incoming gradient $g$. 
  * We broadcast $g$ to create an array of $g$'s the same size as $x$.
* **Max (`np.max`):** Only the highest number actually affected the output. Changing the smaller numbers does absolutely nothing to the maximum! 
  * Therefore, we route the entire incoming gradient $g$ *only to the index that was the maximum*, and give `0` to everything else.

### D. What is `unbroadcast`?
NumPy has a magical feature called "broadcasting." If you add a shape `(3,)` array to a shape `(3, 3)` array, NumPy pretends the smaller array is stretched to `(3, 3)`.
However, during the backward pass, we can't return a gradient of shape `(3, 3)` for an input that was originally `(3,)`. We have to "un-stretch" it by summing the gradients along the axes that were duplicated. That is what the `unbroadcast(g, original_shape)` function does!

### E. Neural Network Specifics
* **ReLU:** Keeps positive numbers, turns negative numbers to 0. 
  * VJP: If $x > 0$, pass the gradient $g$ through. If $x \le 0$, the gradient becomes $0$.
  * Rule: `g * (x > 0)`
* **Convolutions (`conv2d`):** 
  * The VJP for a convolution is actually... another convolution! To find the gradient for the input image, we convolve the incoming gradient $g$ with the *kernel rotated by 180 degrees*.

---

## 4. Summary

Every time you see a rule in `VJP_RULES`, just remember:
1. **`g`** is the "blame" passed down from the final Loss.
2. The rest of the math is just applying standard calculus derivatives, scaled by `g`.
3. We use matrix math/vectors instead of loops so that the GPU/CPU can compute it incredibly fast.
4. We avoid building massive Jacobian matrices, saving huge amounts of memory.

And that is the magic of Vector-Jacobian Products!
