"""
Embedding layers.
"""

from .base import Module
from dnp.core.tensor import Tensor
from dnp.core import ops
from dnp.core.backend import backend


class Embedding(Module):
    """Embedding lookup layer with efficient direct-index slicing."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, name: str = "Embedding"
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.W = self.add_weight(
            "W",
            shape=(num_embeddings, embedding_dim),
            initializer=lambda shape: (
                backend.random.randn(*shape) * backend.sqrt(1.0 / shape[0])
            ),
        )

    def forward(self, x_idx):
        return ops.gather(self.W, x_idx)
