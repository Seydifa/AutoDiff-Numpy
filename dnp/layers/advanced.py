"""
Advanced neural network layers and algorithms.
"""

import numpy as _np
from .base import Module
from dnp.core.tensor import Tensor
from dnp.core import ops
from dnp.core.backend import backend


class SinkhornTransport(Module):
    """Optimal transport map through computational Sinkhorn iterations."""

    def __init__(self, num_iters: int = 10, epsilon: float = 0.1):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    def forward(self, cost_matrix: Tensor, r: Tensor, c: Tensor) -> Tensor:
        # sinkhorn(a, b, M, reg, num_iters): a=source, b=target, M=cost, reg=regularization
        return ops.sinkhorn(r, c, cost_matrix, self.epsilon, num_iters=self.num_iters)


class NeuralODE(Module):
    """Continuous-depth Neural Network layer mapping inputs through an ODE solver."""

    def __init__(self, odefunc, t0: float, t1: float, steps: int = 10):
        super().__init__()
        self.odefunc = odefunc
        self.t0 = t0
        self.t1 = t1
        self.steps = steps

    def forward(self, y0: Tensor) -> Tensor:
        t_span = Tensor(backend.array([self.t0, self.t1]), name="t_span")
        return ops.neural_ode_solve(y0, t_span, steps=self.steps, odefunc=self.odefunc)


class S4Layer(Module):
    """Structured State Space sequence model block (S4).

    Parameters
    ----------
    d_input : int
        Dimensionality of each input token u_t.
    d_model : int
        Internal state-space dimensionality.
    d_output : int
        Dimensionality of each output token y_t.
    """

    def __init__(self, d_input: int, d_model: int, d_output: int):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output

        std = float(backend.sqrt(1.0 / d_model))
        self.A = self.add_weight(
            "A",
            shape=(d_model, d_model),
            initializer=lambda s: backend.random.randn(*s) * std,
        )
        self.B = self.add_weight(
            "B",
            shape=(d_input, d_model),
            initializer=lambda s: backend.random.randn(*s) * std,
        )
        self.C = self.add_weight(
            "C",
            shape=(d_model, d_output),
            initializer=lambda s: backend.random.randn(*s) * std,
        )

    def forward(self, u: Tensor) -> Tensor:
        """Compute S4 output sequence.

        Parameters
        ----------
        u : Tensor, shape (batch, seq_len, d_input)
            Input sequence.

        Returns
        -------
        Tensor, shape (batch, seq_len, d_output)
        """
        return ops.s4_scan(u, self.A, self.B, self.C)


class RNN(Module):
    """Simple Elman Recurrent Neural Network layer."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        std = float(backend.sqrt(1.0 / hidden_size))
        self.Wx = self.add_weight(
            "Wx",
            shape=(input_size, hidden_size),
            initializer=lambda s: backend.random.randn(*s) * std,
        )
        self.Wh = self.add_weight(
            "Wh",
            shape=(hidden_size, hidden_size),
            initializer=lambda s: backend.random.randn(*s) * std,
        )
        self.bh = self.add_weight("bh", shape=(hidden_size,), initializer="zeros")

    def forward(self, x: Tensor, h0: Tensor = None) -> Tensor:
        return ops.rnn_cell(x, self.Wh, self.Wx, self.bh, h0=h0)


class LSTM(Module):
    """Long Short-Term Memory recurrent layer."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        std = float(backend.sqrt(1.0 / hidden_size))
        self.W = self.add_weight(
            "W",
            shape=(hidden_size + input_size, 4 * hidden_size),
            initializer=lambda s: backend.random.randn(*s) * std,
        )
        self.b = self.add_weight("b", shape=(4 * hidden_size,), initializer="zeros")

    def forward(self, x: Tensor, h0: Tensor = None, c0: Tensor = None) -> Tensor:
        return ops.lstm_cell(x, self.W, self.b, h0=h0, c0=c0)


class GRU(Module):
    """Gated Recurrent Unit layer."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        std = float(backend.sqrt(1.0 / hidden_size))
        self.Wr = self.add_weight(
            "Wr",
            shape=(hidden_size + input_size, hidden_size),
            initializer=lambda s: backend.random.randn(*s) * std,
        )
        self.Wz = self.add_weight(
            "Wz",
            shape=(hidden_size + input_size, hidden_size),
            initializer=lambda s: backend.random.randn(*s) * std,
        )
        self.Wh = self.add_weight(
            "Wh",
            shape=(hidden_size + input_size, hidden_size),
            initializer=lambda s: backend.random.randn(*s) * std,
        )

        self.br = self.add_weight("br", shape=(hidden_size,), initializer="zeros")
        self.bz = self.add_weight("bz", shape=(hidden_size,), initializer="zeros")
        self.bh = self.add_weight("bh", shape=(hidden_size,), initializer="zeros")

    def forward(self, x: Tensor, h0: Tensor = None) -> Tensor:
        return ops.gru_cell(
            x, self.Wr, self.Wz, self.Wh, self.br, self.bz, self.bh, h0=h0
        )
