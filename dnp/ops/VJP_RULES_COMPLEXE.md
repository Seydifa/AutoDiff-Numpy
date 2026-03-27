# VJP Rules for Advanced and Complex Layers

This document is the **third and final part** of our Vector-Jacobian Product derivation series, serving as an advanced companion to `VJP_RULES.md` (basic ops) and `VJP_RULES_RECURRENT.md` (standard RNNs and basic attention). 

Here, we provide the rigorous mathematical VJP derivations for **state-of-the-art, highly complex differentiable operations** used in modern deep learning research. These operations often abandon simple step-by-step chain rules in favor of implicit differentiation, continuous adjoint methods, or memory-optimized tiling.

---

## Table of Contents

1. [Neural Ordinary Differential Equations (Neural ODEs)](#1-neural-ordinary-differential-equations-neural-odes)
2. [Structured State Space Models (S4 / Mamba)](#2-structured-state-space-models-s4--mamba)
3. [Rotary Position Embedding (RoPE)](#3-rotary-position-embedding-rope)
4. [FlashAttention (Hardware-Aware VJP)](#4-flashattention-hardware-aware-vjp)
5. [Sinkhorn Optimal Transport (Implicit Differentiation)](#5-sinkhorn-optimal-transport-implicit-differentiation)

---

## 1. Neural Ordinary Differential Equations (Neural ODEs)

Rather than discrete layers, a Neural ODE defines the hidden state continuously using an Ordinary Differential Equation.

### 1.1 Forward Pass

Given an initial state $z(t_0)$ and a neural network $f(z(t), t, \theta)$ parameterized by $\theta$:

$$
z(t_1) = z(t_0) + \int_{t_0}^{t_1} f(z(t), t, \theta) dt
$$

Implemented computationally via a black-box ODE solver: $z(t_1) = \text{ODESolve}(z(t_0), f, t_0, t_1, \theta)$.

### 1.2 Backward Pass (The Continuous Adjoint Method)

Backpropagating through the internal operations of the ODE solver is memory-intensive and numerically unstable. Instead, we define the **adjoint state**:
$$ a(t) = \frac{\partial \mathcal{L}}{\partial z(t)} $$

Given $g = a(t_1)$ from the loss, the VJP for the initial state $z(t_0)$ and parameters $\theta$ are found by solving another ODE **backwards in time**:

$$
\frac{da(t)}{dt} = -a(t)^T \frac{\partial f(z(t), t, \theta)}{\partial z}
$$

$$
\frac{d\mathcal{L}}{d\theta} = -\int_{t_1}^{t_0} a(t)^T \frac{\partial f(z(t), t, \theta)}{\partial \theta} dt
$$

**VJP Rule:** We construct an augmented state $[z(t), a(t), \frac{\partial \mathcal{L}}{\partial \theta}]$ and call the ODE solver *once* running from $t_1$ to $t_0$ to compute the exact gradients in $O(1)$ memory.

---

## 2. Structured State Space Models (S4 / Mamba)

State Space Models (SSMs) treat sequences as samples from a continuous linear dynamical system.

### 2.1 Forward Pass (Discrete Form)

Given matrices $\bar{A}, \bar{B}, \bar{C}$ derived from continuous parameters via Zero-Order Hold (ZOH) discretization:

$$
x_t = \bar{A} x_{t-1} + \bar{B} u_t
$$
$$
y_t = \bar{C} x_t
$$

For S4, this is computed **non-recurrently** using a global convolution:
$$ y = K * u \qquad \text{where } K_t = \bar{C} \bar{A}^t \bar{B} $$

### 2.2 Backward Pass

Given $g_y \in \mathbb{R}^{T \times d_{\text{out}}}$:

**Step 1: Gradient to $C$**
$$
g_{\bar{C}} = \sum_{t} g_{y_t} x_t^T
$$

**Step 2: Adjoint State Recursion**
Just like an RNN, we propagate an adjoint variable $g_{x_t}$ backward:
$$
g_{x_t} = \bar{C}^T g_{y_t} + \bar{A}^T g_{x_{t+1}}
$$

**Step 3: Gradients to $A$ and $B$ (Hardware-Aware)**
$$
g_{\bar{B}} = \sum_t g_{x_t} u_t^T
$$
$$
g_{\bar{A}} = \sum_t g_{x_t} x_{t-1}^T
$$

In modern hardware-aware implementations (like Mamba), this VJP is implemented as a parallel associative scan rather than a sequential loop, allowing gradients to be computed in $O(\log T)$ parallel steps.

---

## 3. Rotary Position Embedding (RoPE)

RoPE encodes relative positional information by rotating pairs of coordinates in the latent space.

### 3.1 Forward Pass

For a vector $x = [x_1, x_2, \ldots, x_d]$, RoPE applies a 2D rotation to adjacent pairs $(x_{2i-1}, x_{2i})$ by an angle $m\theta_i$ (where $m$ is the sequence position):

$$
\begin{bmatrix} y_{2i-1} \\ y_{2i} \end{bmatrix} =
\begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}
\begin{bmatrix} x_{2i-1} \\ x_{2i} \end{bmatrix}
$$

### 3.2 Backward Pass

The Jacobian of a rotation matrix $R(\theta)$ is simply the rotation matrix itself. Because $R(\theta)$ is orthogonal, its transpose is its inverse:  $R(\theta)^T = R(-\theta)$.

Therefore, the VJP of RoPE is incredibly simple: **it is simply RoPE applied with the negative angle.**

$$
\boxed{g_x = \text{RoPE}(g_y, -m\theta)}
$$

Because the angles are fixed frequencies, no gradient flows to $\theta$.

---

## 4. FlashAttention (Hardware-Aware VJP)

Standard attention requires materializing the $N \times N$ attention matrix $S = QK^T$, dominating memory. FlashAttention fuses the operation.

### 4.1 Forward Pass

Block-tiled exact attention computation utilizing SRAM:
$$ O = \text{softmax}(QK^T)V $$

### 4.2 Backward Pass

Standard attention VJP requires saving the massive $N \times N$ matrix $P = \text{softmax}(S)$ to compute the softmax backward pass:
$$ g_S = P \odot (g_P - \text{rowsum}(P \odot g_P)) $$

**FlashAttention VJP Bypass:**
Instead of saving $P$, FlashAttention **recomputes** $P_{ij}$ on the fly during the backward pass backward, tile-by-tile in SRAM.

Let $D_i = \sum_j (g_{O_{i,:}} \odot O_{i,:})$ be the row-wise dot product (saved from the forward pass in $O(N)$ memory).
For a specific block of $Q, K, V$:

$$
dP_{ij} = g_{O_i} V_j^T
$$
$$
dS_{ij} = P_{ij} (dP_{ij} - D_i)
$$
$$
dQ_i \mathrel{+}= dS_{ij} K_j, \qquad dK_j \mathrel{+}= dS_{ij}^T Q_i, \qquad dV_j \mathrel{+}= P_{ij}^T g_{O_i}
$$

**VJP Insight:** The mathematical rule is mathematically identical to standard attention (see `VJP_RULES_RECURRENT.md`), but the **dependency graph is manually rewritten** to rely on recomputation, trading a small amount of compute for a massive $O(N^2) \to O(N)$ memory reduction in the backward pass.

---

## 5. Sinkhorn Optimal Transport (Implicit Differentiation)

Used in matching networks and optimal transport, Sinkhorn iterations find a doubly-stochastic assignment matrix.

### 5.1 Forward Pass

Given a cost matrix $M$, we want $P = \arg\min_P \langle P, M \rangle - \epsilon H(P)$.
We unroll Sinkhorn iterations:
$K = \exp(-M/\epsilon)$
Iterate: $u \leftarrow 1 / (K v)$, $v \leftarrow 1 / (K^T u)$
Output: $P = \text{diag}(u) K \text{diag}(v)$

### 5.2 Backward Pass

Backpropagating explicitly through the unrolled `while` loop requires storing every intermediate $u$ and $v$ variable, causing massive memory overhead.

Instead, we use the **Implicit Function Theorem**. At the fixed point, the gradients rely *only* on the final $(u, v)$ solution.

Given $g_P = \frac{\partial \mathcal{L}}{\partial P}$:
1. Compute the adjoint vectors using the final fixed point.
2. The VJP to the cost matrix $M$ is computed instantly in a single step without traversing the loop history.

$$
g_M = -\frac{1}{\epsilon} P \odot (g_P - \alpha \mathbf{1}^T - \mathbf{1} \beta^T)
$$

where $\alpha$ and $\beta$ are dual variables obtained by solving an implicit linear system defined by the final state of $P$.

---
*By structuring VJPs cleanly using adjoints or implicit theorem, Deep Learning models can handle computationally infinite forward passes while keeping the backward pass exact and in constant memory!*
