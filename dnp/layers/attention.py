"""
Attention mechanisms and Transformer blocks.
"""

import math
from .base import Module
from dnp.layers.linear import Linear
from dnp.layers.normalization import LayerNorm
from dnp.layers.regularization import Dropout
from dnp.core.tensor import Tensor
from dnp.core.backend import backend, get_dtype, is_cupy_array, to_device
from dnp.core import ops
import numpy as _np


class ScaledDotProductAttention(Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.scale = float(1.0 / math.sqrt(d_k)) if d_k else None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None) -> Tensor:
        return ops.scaled_dot_product_attention(query, key, value, mask=mask)


class MultiHeadAttention(Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.W_o = Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None) -> Tensor:
        batch_size, seq_len = query.shape[:2]

        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        q = ops.reshape(q, newshape=(batch_size, seq_len, self.num_heads, self.d_k))
        k = ops.reshape(k, newshape=(batch_size, seq_len, self.num_heads, self.d_k))
        v = ops.reshape(v, newshape=(batch_size, seq_len, self.num_heads, self.d_k))

        q = ops.transpose(q, axes=(0, 2, 1, 3))
        k = ops.transpose(k, axes=(0, 2, 1, 3))
        v = ops.transpose(v, axes=(0, 2, 1, 3))

        attn_output = self.attention(q, k, v, mask)

        attn_output = ops.transpose(attn_output, axes=(0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output, newshape=(batch_size, seq_len, self.d_model)
        )

        output = self.W_o(attn_output)
        return output


class SelfAttention(Module):
    """Self-attention layer (query, key, value all from same source)."""

    def __init__(self, d_model: int, num_heads: int = 1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        return self.mha(x, x, x, mask)


class PositionalEncoding(Module):
    """Sinusoidal positional encoding (non-trainable)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout_p: float = 0.1):
        super().__init__()
        self.dropout_p = dropout_p

        position = backend.arange(max_len)[:, None]
        div_term = backend.exp(
            backend.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = backend.zeros((max_len, d_model), dtype=get_dtype())
        pe[:, 0::2] = backend.sin(position * div_term)
        pe[:, 1::2] = backend.cos(position * div_term)
        self._buffers["pe"] = pe[None, :, :]

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        pe = self._buffers["pe"]
        pe_device = "cuda" if is_cupy_array(pe) else "cpu"
        if pe_device != x.device:
            pe = to_device(pe, x.device)
            self._buffers["pe"] = pe
        pe_slice = Tensor(pe[:, :seq_len, :])
        x = x + pe_slice
        if self.training and self.dropout_p > 0.0:
            x = ops.dropout(x, p=self.dropout_p, training=True)
        return x


class FeedForward(Module):
    """Position-wise Feed-Forward block used in transformer layers."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "relu",
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout_p = dropout_p

        _act_map = {"relu": ops.relu, "gelu": ops.gelu, "swish": ops.swish}
        if activation not in _act_map:
            raise ValueError(f"Unsupported activation '{activation}'.")
        self._act_fn = _act_map[activation]

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self._act_fn(x)
        if self.training and self.dropout_p > 0.0:
            x = ops.dropout(x, p=self.dropout_p, training=True)
        return self.fc2(x)


class TransformerEncoderLayer(Module):
    """Single transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_p: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model, d_ff, activation=activation, dropout_p=dropout_p
        )
        self.norm2 = LayerNorm(d_model)
        self.dropout_p = dropout_p

    def forward(self, x: Tensor, mask=None) -> Tensor:
        attn_out = self.attn(x, x, x, mask)
        if self.training and self.dropout_p > 0.0:
            attn_out = ops.dropout(attn_out, p=self.dropout_p, training=True)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TransformerDecoderLayer(Module):
    """Single transformer decoder layer.

    Implements the standard three-sublayer decoder block:
    1. Masked self-attention  (query = key = value = target sequence)
    2. Cross-attention        (query = target, key/value = encoder output)
    3. Position-wise FFN

    Each sublayer is wrapped with Add & Norm (pre-norm style is not used;
    the residual is added *before* the layer norm for compatibility with the
    original "Attention is All You Need" paper).

    Parameters
    ----------
    d_model : int
        Model / embedding dimensionality.
    num_heads : int
        Number of parallel attention heads.  Must divide ``d_model``.
    d_ff : int
        Inner dimensionality of the position-wise feed-forward network.
    dropout_p : float, default 0.1
        Dropout probability applied after each sub-layer (only during training).
    activation : str, default 'relu'
        Activation inside the feed-forward network ('relu' or 'gelu').
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_p: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        # 1. Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        # 2. Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        # 3. FFN
        self.ffn = FeedForward(
            d_model, d_ff, activation=activation, dropout_p=dropout_p
        )
        self.norm3 = LayerNorm(d_model)
        self.dropout_p = dropout_p

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask=None,
        memory_mask=None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        tgt : Tensor, shape (batch, tgt_len, d_model)
            Target sequence (decoder input).
        memory : Tensor, shape (batch, src_len, d_model)
            Encoder output.
        tgt_mask : array-like or None
            Optional mask for the self-attention sublayer.
        memory_mask : array-like or None
            Optional mask for the cross-attention sublayer.

        Returns
        -------
        Tensor, shape (batch, tgt_len, d_model)
        """
        # --- sublayer 1: masked self-attention ---
        sa_out = self.self_attn(tgt, tgt, tgt, tgt_mask)
        if self.training and self.dropout_p > 0.0:
            sa_out = ops.dropout(sa_out, p=self.dropout_p, training=True)
        tgt = self.norm1(tgt + sa_out)

        # --- sublayer 2: cross-attention ---
        ca_out = self.cross_attn(tgt, memory, memory, memory_mask)
        if self.training and self.dropout_p > 0.0:
            ca_out = ops.dropout(ca_out, p=self.dropout_p, training=True)
        tgt = self.norm2(tgt + ca_out)

        # --- sublayer 3: feed-forward ---
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + ffn_out)
        return tgt


# -----------------------------------------------------------------
# COMPLEX / ADVANCED ATTENTION
# -----------------------------------------------------------------


class FlashAttention(Module):
    """Memory-efficient exact attention mechanism."""

    def __init__(self, dropout_p: float = 0.0, is_causal: bool = False):
        super().__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
        return ops.flash_attention(q, k, v, mask=mask)


class RotaryPositionalEncoding(Module):
    """RoPE (Rotary Position Embedding)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

    def forward(self, x: Tensor, seq_dim: int = 1) -> Tensor:
        seq_len = x.shape[seq_dim]
        half_d = self.dim // 2
        dtype = get_dtype()
        # theta_i = base^(-2i/dim) for i = 0, ..., half_d - 1
        theta = _np.power(
            float(self.base),
            -_np.arange(0, half_d, dtype=_np.float32) * (2.0 / self.dim),
        ).astype(dtype)
        positions = _np.arange(seq_len, dtype=dtype)
        freqs = _np.outer(positions, theta)  # (seq_len, half_d)
        cos_freqs = Tensor(backend.asarray(_np.cos(freqs)), name="rope_cos")
        sin_freqs = Tensor(backend.asarray(_np.sin(freqs)), name="rope_sin")
        return ops.rope(x, cos_freqs, sin_freqs)
