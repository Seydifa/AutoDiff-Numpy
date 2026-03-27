# VJP Rules for Recurrent and Sequence Layers

This document provides the complete, rigorous mathematical derivations of
**Vector-Jacobian Products (VJPs)** for every major recurrent and sequence-modelling
layer. It complements `VJP_RULES.md` (which covers element-wise ops, convolutions, and
pooling) and is written to be the primary reference for implementing `backward()` in
`dnp/core/vjp_rules.py`.

> **Convention** used throughout:
> - $g_y = \dfrac{\partial L}{\partial y}$ denotes the incoming gradient from the loss to output $y$.
> - $\odot$ is the Hadamard (element-wise) product.
> - $[a\,;\,b]$ denotes row-wise concatenation along the feature axis.
> - Subscript $t$ indexes time steps; $T$ is the sequence length.
> - All weight gradients **accumulate** (+=) over time steps (BPTT).

---

## Table of Contents

1. [Simple (Elman) RNN](#1-simple-elman-rnn)
2. [LSTM — Long Short-Term Memory](#2-lstm)
3. [GRU — Gated Recurrent Unit](#3-gru)
4. [Peephole LSTM](#4-peephole-lstm)
5. [Bidirectional Wrappers](#5-bidirectional-wrappers)
6. [Scaled Dot-Product Attention](#6-scaled-dot-product-attention)
7. [Multi-Head Attention](#7-multi-head-attention)
8. [Layer Normalization](#8-layer-normalization)
9. [Embedding Layer](#9-embedding-layer)
10. [Connectionist Temporal Classification (CTC) Loss](#10-ctc-loss)
11. [Summary Table](#11-summary-table)

---

## 1. Simple (Elman) RNN

### 1.1 Forward Pass

At each time step $t \in \{1, \ldots, T\}$:

$$
s_t = W_h\, h_{t-1} + W_x\, x_t + b_h \qquad \text{(pre-activation)}
$$

$$
h_t = \tanh(s_t)
$$

$$
y_t = W_y\, h_t + b_y \qquad \text{(optional output projection)}
$$

**Shapes:** $x_t \in \mathbb{R}^{B \times d_{\text{in}}}$, $h_t \in \mathbb{R}^{B \times d_h}$,
$W_h \in \mathbb{R}^{d_h \times d_h}$, $W_x \in \mathbb{R}^{d_{\text{in}} \times d_h}$,
$b_h \in \mathbb{R}^{d_h}$.

### 1.2 Backward Pass (BPTT)

At each step $t$ (processed in **reverse order** $T, T{-}1, \ldots, 1$) we have the
accumulated incoming gradient:

$$
g_{h_t} = \underbrace{g_{h_t}^{\text{output}}}_{\text{from loss at } t}
         + \underbrace{g_{h_t}^{\text{next}}}_{\text{from step }t+1}
$$

**Step 1 — backprop through tanh:**

$$
\boxed{\delta_t = g_{h_t} \odot \bigl(1 - h_t^2\bigr)}
$$

**Step 2 — weight gradients (accumulated):**

$$
g_{W_h} \mathrel{+}= h_{t-1}^{\top}\, \delta_t
$$

$$
g_{W_x} \mathrel{+}= x_t^{\top}\, \delta_t
$$

$$
g_{b_h} \mathrel{+}= \sum_{\text{batch}} \delta_t
$$

**Step 3 — pass gradient to previous step and input:**

$$
\boxed{g_{h_{t-1}} = \delta_t\, W_h^{\top}}
$$

$$
\boxed{g_{x_t} = \delta_t\, W_x^{\top}}
$$

**Output projection (if used):**

$$
g_{W_y} \mathrel{+}= h_t^{\top}\, g_{y_t}, \quad
g_{b_y} \mathrel{+}= \sum_{\text{batch}} g_{y_t}, \quad
g_{h_t}^{\text{output}} = g_{y_t}\, W_y^{\top}
$$

---

## 2. LSTM

### 2.1 Forward Pass

Define the combined input $z_t = [h_{t-1}\,;\,x_t] \in \mathbb{R}^{B \times (d_h + d_{\text{in}})}$.
A single weight matrix $W \in \mathbb{R}^{(d_h+d_{\text{in}}) \times 4d_h}$ and bias
$b \in \mathbb{R}^{4d_h}$ handle all four gates simultaneously.

$$
[f_t,\, i_t,\, \tilde{c}_t,\, o_t]
= \bigl[\sigma,\; \sigma,\; \tanh,\; \sigma\bigr]\!\left(z_t\, W + b\right)
$$

Or spelled out individually:

$$
f_t = \sigma\!\left(W_f z_t + b_f\right) \qquad \text{(forget gate)}
$$

$$
i_t = \sigma\!\left(W_i z_t + b_i\right) \qquad \text{(input gate)}
$$

$$
\tilde{c}_t = \tanh\!\left(W_c z_t + b_c\right) \qquad \text{(cell candidate)}
$$

$$
o_t = \sigma\!\left(W_o z_t + b_o\right) \qquad \text{(output gate)}
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \qquad \text{(cell state update)}
$$

$$
h_t = o_t \odot \tanh(c_t) \qquad \text{(hidden state)}
$$

### 2.2 Backward Pass (BPTT)

At each step $t$ we receive:
- $g_{h_t}$ — gradient from the loss / from step $t+1$.
- $g_{c_t}$ — gradient flowing through the cell state from step $t+1$.

**Step 1 — output gate and cell state gradient:**

$$
d_{o_t} = g_{h_t} \odot \tanh(c_t)
$$

$$
\boxed{g_{c_t}^{\text{total}} = g_{c_t} + g_{h_t} \odot o_t \odot \bigl(1 - \tanh^2(c_t)\bigr)}
$$

**Step 2 — gate error signals from cell state:**

$$
d_{f_t} = g_{c_t}^{\text{total}} \odot c_{t-1}
$$

$$
d_{i_t} = g_{c_t}^{\text{total}} \odot \tilde{c}_t
$$

$$
d_{\tilde{c}_t} = g_{c_t}^{\text{total}} \odot i_t
$$

**Step 3 — gradient to previous cell state:**

$$
\boxed{g_{c_{t-1}} = g_{c_t}^{\text{total}} \odot f_t}
$$

**Step 4 — pre-activation (sigmoid/tanh) derivatives:**

$$
\delta_{f_t} = d_{f_t} \odot f_t \odot (1 - f_t)
$$

$$
\delta_{i_t} = d_{i_t} \odot i_t \odot (1 - i_t)
$$

$$
\delta_{\tilde{c}_t} = d_{\tilde{c}_t} \odot (1 - \tilde{c}_t^2)
$$

$$
\delta_{o_t} = d_{o_t} \odot o_t \odot (1 - o_t)
$$

**Step 5 — concatenated gate delta and weight gradients:**

$$
\Delta_t = [\delta_{f_t},\; \delta_{i_t},\; \delta_{\tilde{c}_t},\; \delta_{o_t}]
\quad \in \mathbb{R}^{B \times 4d_h}
$$

$$
g_W \mathrel{+}= z_t^{\top}\, \Delta_t
$$

$$
g_b \mathrel{+}= \sum_{\text{batch}} \Delta_t
$$

**Step 6 — gradient to combined input, then split:**

$$
g_{z_t} = \Delta_t\, W^{\top}
$$

$$
\boxed{g_{h_{t-1}} = g_{z_t}[\,:\,,\, :d_h] , \qquad g_{x_t} = g_{z_t}[\,:\,,\, d_h:]}
$$

> **Memory-efficient implementation:** store $f_t, i_t, \tilde{c}_t, o_t, c_t, h_t$ for
> every $t$ during the forward pass (the "cache"). In truncated BPTT, only a window of
> steps is stored.

---

## 3. GRU

### 3.1 Forward Pass

$$
r_t = \sigma\!\left(W_r [h_{t-1}\,;\,x_t] + b_r\right) \qquad \text{(reset gate)}
$$

$$
z_t = \sigma\!\left(W_z [h_{t-1}\,;\,x_t] + b_z\right) \qquad \text{(update gate)}
$$

$$
\tilde{h}_t = \tanh\!\left(W_h [r_t \odot h_{t-1}\,;\,x_t] + b_h\right)
\qquad \text{(candidate hidden)}
$$

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \qquad \text{(hidden update)}
$$

> There is also the **PyTorch-style GRU** which places the reset inside the candidate:
> $\tilde{h}_t = \tanh\!\bigl(r_t \odot (W_{hh} h_{t-1}) + W_{xh} x_t + b_h\bigr)$.
> The VJP derivation structure is identical; only the input to the candidate changes.

### 3.2 Backward Pass (BPTT)

At each step $t$ we receive $g_{h_t}$.

**Step 1 — direct gradients from hidden update equation:**

$$
d_{\tilde{h}_t} = g_{h_t} \odot z_t
$$

$$
d_{z_t} = g_{h_t} \odot (\tilde{h}_t - h_{t-1})
$$

$$
g_{h_{t-1}}^{(1)} = g_{h_t} \odot (1 - z_t) \qquad \text{(direct passthrough)}
$$

**Step 2 — backprop through candidate $\tilde{h}_t$:**

$$
\delta_{\tilde{h}_t} = d_{\tilde{h}_t} \odot (1 - \tilde{h}_t^2)
$$

$$
g_{W_h} \mathrel{+}= [r_t \odot h_{t-1}\,;\,x_t]^{\top}\, \delta_{\tilde{h}_t}
$$

$$
g_{b_h} \mathrel{+}= \sum_{\text{batch}} \delta_{\tilde{h}_t}
$$

$$
[d_{r \odot h}\,;\, g_{x_t}^{(1)}] = \delta_{\tilde{h}_t}\, W_h^{\top}
$$

**Step 3 — backprop through reset gate multiplication:**

$$
d_{r_t} = d_{r \odot h} \odot h_{t-1}
$$

$$
g_{h_{t-1}}^{(2)} = d_{r \odot h} \odot r_t
$$

**Step 4 — backprop through reset gate sigmoid:**

$$
\delta_{r_t} = d_{r_t} \odot r_t \odot (1 - r_t)
$$

$$
g_{W_r} \mathrel{+}= [h_{t-1}\,;\,x_t]^{\top}\, \delta_{r_t}
$$

$$
g_{b_r} \mathrel{+}= \sum_{\text{batch}} \delta_{r_t}
$$

$$
[g_{h_{t-1}}^{(3)}\,;\, g_{x_t}^{(2)}] = \delta_{r_t}\, W_r^{\top}
$$

**Step 5 — backprop through update gate sigmoid:**

$$
\delta_{z_t} = d_{z_t} \odot z_t \odot (1 - z_t)
$$

$$
g_{W_z} \mathrel{+}= [h_{t-1}\,;\,x_t]^{\top}\, \delta_{z_t}
$$

$$
g_{b_z} \mathrel{+}= \sum_{\text{batch}} \delta_{z_t}
$$

$$
[g_{h_{t-1}}^{(4)}\,;\, g_{x_t}^{(3)}] = \delta_{z_t}\, W_z^{\top}
$$

**Step 6 — sum all partial gradients:**

$$
\boxed{g_{h_{t-1}} = g_{h_{t-1}}^{(1)} + g_{h_{t-1}}^{(2)} + g_{h_{t-1}}^{(3)} + g_{h_{t-1}}^{(4)}}
$$

$$
\boxed{g_{x_t} = g_{x_t}^{(1)} + g_{x_t}^{(2)} + g_{x_t}^{(3)}}
$$

---

## 4. Peephole LSTM

Peephole connections let the gates "look at" the cell state directly.

### 4.1 Forward Pass

$$
f_t = \sigma\!\left(W_f [h_{t-1}\,;\,x_t] + p_f \odot c_{t-1} + b_f\right)
$$

$$
i_t = \sigma\!\left(W_i [h_{t-1}\,;\,x_t] + p_i \odot c_{t-1} + b_i\right)
$$

$$
\tilde{c}_t = \tanh\!\left(W_c [h_{t-1}\,;\,x_t] + b_c\right)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

$$
o_t = \sigma\!\left(W_o [h_{t-1}\,;\,x_t] + p_o \odot c_t + b_o\right)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

where $p_f, p_i, p_o \in \mathbb{R}^{d_h}$ are the peephole weight vectors.

### 4.2 Backward Pass

The standard LSTM VJP applies, with the following additions for peephole weights.

From the output gate: $d_{o_t} = g_{h_t} \odot \tanh(c_t)$, then

$$
\delta_{o_t} = d_{o_t} \odot o_t \odot (1 - o_t)
$$

$$
g_{p_o} \mathrel{+}= \sum_{\text{batch}} \delta_{o_t} \odot c_t
$$

$$
g_{c_t}^{\text{peep-o}} = \delta_{o_t} \odot p_o
$$

The total cell gradient is:

$$
g_{c_t}^{\text{total}} = g_{c_t} + g_{h_t} \odot o_t \odot (1 - \tanh^2 c_t) + g_{c_t}^{\text{peep-o}}
$$

From the forget and input gates:

$$
g_{p_f} \mathrel{+}= \sum_{\text{batch}} \delta_{f_t} \odot c_{t-1}, \qquad
g_{p_i} \mathrel{+}= \sum_{\text{batch}} \delta_{i_t} \odot c_{t-1}
$$

$$
g_{c_{t-1}} = g_{c_t}^{\text{total}} \odot f_t + \delta_{f_t} \odot p_f + \delta_{i_t} \odot p_i
$$

All other steps (weight gradients, $g_{h_{t-1}}$, $g_{x_t}$) are identical to the
standard LSTM.

---

## 5. Bidirectional Wrappers

A bidirectional layer runs a **forward cell** ($\overrightarrow{\text{cell}}$) left-to-right
and a **backward cell** ($\overleftarrow{\text{cell}}$) right-to-left, then concatenates
the hidden states at each time.

### 5.1 Forward Pass

$$
\overrightarrow{h}_t = \overrightarrow{\text{cell}}(x_t,\, \overrightarrow{h}_{t-1})
\qquad t = 1, \ldots, T
$$

$$
\overleftarrow{h}_t = \overleftarrow{\text{cell}}(x_t,\, \overleftarrow{h}_{t+1})
\qquad t = T, \ldots, 1
$$

$$
h_t = [\overrightarrow{h}_t\,;\, \overleftarrow{h}_t]
$$

### 5.2 Backward Pass

Split the incoming gradient by direction:

$$
g_{\overrightarrow{h}_t} = g_{h_t}[\,:\,,\, :d_h], \qquad
g_{\overleftarrow{h}_t} = g_{h_t}[\,:\,,\, d_h:]
$$

Run the standard single-direction VJP for each cell independently:

- Forward cell: process $t = T, \ldots, 1$ using $g_{\overrightarrow{h}_t}$.
- Backward cell: process $t = 1, \ldots, T$ using $g_{\overleftarrow{h}_t}$.

The gradient w.r.t. $x_t$ is the **sum** of contributions from both directions:

$$
\boxed{g_{x_t} = g_{x_t}^{\overrightarrow{\text{cell}}} + g_{x_t}^{\overleftarrow{\text{cell}}}}
$$

---

## 6. Scaled Dot-Product Attention

### 6.1 Forward Pass

$$
E = \frac{Q K^{\top}}{\sqrt{d_k}}
\qquad Q \in \mathbb{R}^{B \times N \times d_k},\;
  K \in \mathbb{R}^{B \times M \times d_k},\;
  V \in \mathbb{R}^{B \times M \times d_v}
$$

$$
A = \text{softmax}(E) \in \mathbb{R}^{B \times N \times M}
\qquad A_{ij} = \frac{\exp E_{ij}}{\sum_k \exp E_{ik}}
$$

$$
\text{Attn} = A V \in \mathbb{R}^{B \times N \times d_v}
$$

**Masked attention:** before softmax, set $E_{ij} \leftarrow -\infty$ for masked positions.
The VJP is unaffected (masked positions have $A_{ij} = 0$ and $g_{E_{ij}}$ is naturally 0).

### 6.2 Backward Pass

Given $g = \dfrac{\partial L}{\partial \text{Attn}}$:

**Step 1 — gradient to $V$ and $A$:**

$$
\boxed{g_V = A^{\top} g} \qquad (M \times d_v)
$$

$$
g_A = g\, V^{\top} \qquad (N \times M)
$$

**Step 2 — softmax VJP:**

For each row $i$ of the attention matrix let $a = A_{i,:}$, $u = g_{A_{i,:}}$:

$$
\boxed{g_{E_{i,:}} = a \odot \bigl(u - \langle a, u \rangle \mathbf{1}\bigr)}
$$

In matrix form (all rows at once):

$$
g_E = A \odot \Bigl(g_A - \sum_j (A \odot g_A)_j\, \mathbf{1}^{\top}\Bigr)
    = A \odot \Bigl(g_A - (A \odot g_A)\,\mathbf{1}\,\mathbf{1}^{\top}\Bigr)
$$

where the row-wise dot is $\text{rowsum}(A \odot g_A)$, broadcast back to shape $(N \times M)$.

**Step 3 — gradient to $Q$ and $K$:**

$$
\boxed{g_Q = \frac{g_E\, K}{\sqrt{d_k}}}
\qquad
\boxed{g_K = \frac{g_E^{\top} Q}{\sqrt{d_k}}}
$$

---

## 7. Multi-Head Attention

### 7.1 Forward Pass

With $H$ heads, projection matrices $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$
and output projection $W^O \in \mathbb{R}^{H d_v \times d_{\text{model}}}$:

$$
\text{head}_i = \text{Attention}\!\left(Q W_i^Q,\; K W_i^K,\; V W_i^V\right)
$$

$$
\text{MultiHead}(Q,K,V) = [\text{head}_1;\, \ldots;\, \text{head}_H]\, W^O
$$

### 7.2 Backward Pass

Given $g_{\text{out}} \in \mathbb{R}^{B \times N \times d_{\text{model}}}$:

**Step 1 — output projection:**

Let $C = [\text{head}_1;\, \ldots;\, \text{head}_H] \in \mathbb{R}^{B \times N \times Hd_v}$.

$$
g_C = g_{\text{out}}\, (W^O)^{\top}
$$

$$
g_{W^O} = \sum_{\text{batch}} C^{\top} g_{\text{out}}
$$

**Step 2 — split gradient per head and apply attention VJP:**

$$
g_{\text{head}_i} = g_C[\,:\,,\,:\,,\, (i{-}1)d_v : i\,d_v]
$$

Apply the scaled dot-product attention VJP (Section 6.2) per head $i$ to get
$g_{Q_i}, g_{K_i}, g_{V_i}$ where $Q_i = Q W_i^Q$, etc.

**Step 3 — projection-matrix gradients:**

$$
g_{W_i^Q} = \sum_{\text{batch}} Q^{\top} g_{Q_i}, \qquad
g_{W_i^K} = \sum_{\text{batch}} K^{\top} g_{K_i}, \qquad
g_{W_i^V} = \sum_{\text{batch}} V^{\top} g_{V_i}
$$

**Step 4 — input gradients (sum over heads):**

$$
\boxed{g_Q = \sum_{i=1}^{H} g_{Q_i}\, (W_i^Q)^{\top}}
$$

$$
\boxed{g_K = \sum_{i=1}^{H} g_{K_i}\, (W_i^K)^{\top}}
$$

$$
\boxed{g_V = \sum_{i=1}^{H} g_{V_i}\, (W_i^V)^{\top}}
$$

---

## 8. Layer Normalization

### 8.1 Forward Pass

Normalize over the last $d$ axes (typically the feature dimension $D$):

$$
\mu = \frac{1}{D}\sum_{j=1}^{D} x_j
$$

$$
\sigma^2 = \frac{1}{D}\sum_{j=1}^{D}(x_j - \mu)^2
$$

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y = \gamma \odot \hat{x} + \beta
\qquad \gamma,\beta \in \mathbb{R}^{D}
$$

### 8.2 Backward Pass

Given $g = \dfrac{\partial L}{\partial y}$:

**Scale/bias gradients:**

$$
g_\gamma = \sum_{\text{batch/seq}} g \odot \hat{x}
$$

$$
g_\beta = \sum_{\text{batch/seq}} g
$$

**Input gradient** (standard derivation via chain rule through $\hat{x}$, $\sigma^2$, $\mu$):

$$
g_{\hat{x}} = g \odot \gamma
$$

$$
g_{\sigma^2} = -\frac{1}{2}\sum_j g_{\hat{x}_j}(x_j - \mu)(\sigma^2 + \epsilon)^{-3/2}
$$

$$
g_\mu = -\frac{1}{\sqrt{\sigma^2+\epsilon}}\sum_j g_{\hat{x}_j}
         - 2\,g_{\sigma^2}\frac{1}{D}\sum_j (x_j - \mu)
$$

Combining into a single closed-form expression:

$$
\boxed{
g_x = \frac{1}{D\sqrt{\sigma^2+\epsilon}}
\left[
D\, g_{\hat{x}}
- \sum_{j} g_{\hat{x}_j}
- \hat{x}\sum_{j} g_{\hat{x}_j}\hat{x}_j
\right]
}
$$

This formula is valid whether LayerNorm is used standalone or inside a Transformer
pre-norm / post-norm block.

---

## 9. Embedding Layer

### 9.1 Forward Pass

Given a lookup table $E \in \mathbb{R}^{V \times d}$ (vocabulary size $V$, embedding dim $d$)
and integer indices $x \in \mathbb{Z}^{B \times N}$:

$$
y = E[x] \in \mathbb{R}^{B \times N \times d}
$$

### 9.2 Backward Pass

Because $x$ consists of integer indices, **no gradient flows to $x$**
(integer-valued inputs are non-differentiable).

The gradient w.r.t. the embedding matrix $E$ is a **sparse scatter-add**:

$$
\boxed{g_E[k] = \sum_{\{(b,n)\,:\, x_{b,n}=k\}} g_{b,n,:} \qquad \forall k \in \{0,\ldots,V{-}1\}}
$$

In NumPy this is implemented as:

```python
g_E = np.zeros_like(E)
np.add.at(g_E, x, g)   # g has shape (B, N, d)
```

For GPU backends and large vocabularies, a segment-reduce variant is preferred for
numerical stability and performance.

---

## 10. CTC Loss

### 10.1 Forward Pass

Given log-probabilities $\log p \in \mathbb{R}^{T \times B \times C}$ (sequence length $T$,
batch $B$, $C$ classes including blank) and target sequences $y^*$, CTC computes:

$$
\mathcal{L}_{\text{CTC}} = -\sum_b \log P(y^*_b \mid x_b)
$$

using the forward-backward (α-β) algorithm over the lattice of valid alignments.

### 10.2 Backward Pass (Graves 2006)

Let $\alpha_{t,s}$ (forward variable) and $\beta_{t,s}$ (backward variable) be the
standard CTC forward/backward probabilities for position $s$ in the extended label
sequence $\ell'$ (with blanks inserted).

The gradient of the CTC loss w.r.t. the log-softmax output $\log p_{t,k}$ is:

$$
\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial \log p_{t,k}}
= p_{t,k} - \frac{1}{P(y^* \mid x)}
  \sum_{\{s\,:\,\ell'_s = k\}} \alpha_{t,s}\,\beta_{t,s}
$$

In practice this can overflow without log-space computation. The numerically stable
log-space version is:

$$
\boxed{
g_{\log p_{t,k}}
= p_{t,k}
- \exp\!\Bigl(
    \text{logsumexp}_{s:\ell'_s=k}\bigl(\log\alpha_{t,s} + \log\beta_{t,s}\bigr)
    - \log P(y^*\mid x)
  \Bigr)
}
$$

This gradient flows back through the softmax layer in the usual way. No gradient is
defined w.r.t. the target labels $y^*$ (they are not floating-point parameters).

---

## 11. Summary Table

| Layer | Trainable Params | Key VJP Signal(s) | Recurrence |
|---|---|---|---|
| Simple RNN | $W_h, W_x, b_h$ | $\delta_t = g_{h_t} \odot (1 - h_t^2)$ | $g_{h_{t-1}} = \delta_t W_h^\top$ |
| LSTM | $W$ (4 gates), $b$ | $g_{c_t}^{\text{total}}$ splits to 4 gate deltas | $g_{c_{t-1}} = g_{c_t}^{\text{total}} \odot f_t$ |
| GRU | $W_r, W_z, W_h, b_*$ | 4 partial $g_{h_{t-1}}$ summed | No separate cell state |
| Peephole LSTM | + peephole $p_f, p_i, p_o$ | Extra $\odot p_*$ terms in $g_{c_{t-1}}$ | Same as LSTM + peephole correction |
| Bidirectional | 2× cell params | Split $g_{h_t}$ by direction | Two independent BPTT sweeps |
| Dot-Product Attn | — | $g_V = A^\top g$, softmax-VJP for $g_E$ | — |
| Multi-Head Attn | $W_i^{Q,K,V}$, $W^O$ | Sum per-head $g_{Q/K/V}$ | — |
| Layer Norm | $\gamma, \beta$ | Closed-form $g_x$ from $g_{\hat{x}}$ | — |
| Embedding | $E$ | Sparse `np.add.at` scatter | No gradient to indices |
| CTC Loss | — | $g = p - \text{(α⊗β normalised)}$ | — |

---

## References

1. S. Hochreiter & J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 1997.
2. K. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder," *EMNLP*, 2014.
3. F. Gers, J. Schmidhuber & F. Cummins, "Learning to Forget: Continual Prediction with LSTM," *ICANN*, 1999.
4. A. Graves, "Connectionist Temporal Classification," *ICML*, 2006.
5. A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.
6. J. Lei Ba et al., "Layer Normalization," *arXiv:1607.06450*, 2016.
7. R. Pascanu, T. Mikolov & Y. Bengio, "On the Difficulty of Training RNNs," *ICML*, 2013.
8. M. Schuster & K. Paliwal, "Bidirectional Recurrent Neural Networks," *IEEE Trans. Signal Process.*, 1997.
