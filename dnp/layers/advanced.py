"""
Advanced neural network layers and algorithms.
"""
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
        return ops.sinkhorn(
            cost_matrix, r, c, 
            epsilon=self.epsilon, 
            num_iters=self.num_iters
        )


class NeuralODE(Module):
    """Continuous-depth Neural Network layer mapping inputs through an ODE solver."""
    
    def __init__(self, odefunc, t0: float, t1: float, steps: int = 10):
        super().__init__()
        self.odefunc = odefunc
        self.t0 = t0
        self.t1 = t1
        self.steps = steps

    def forward(self, y0: Tensor) -> Tensor:
        return ops.neural_ode_solve(
            self.odefunc, 
            y0, 
            self.t0, 
            self.t1, 
            self.steps
        )


class S4Layer(Module):
    """Structured State Space sequence model block (S4)."""
    
    def forward(self, u: Tensor, B: Tensor, C: Tensor, log_dt: Tensor, delta_A: Tensor) -> Tensor:
        return ops.s4_scan(u, B, C, log_dt, delta_A)


class RNN(Module):
    """Simple Elman Recurrent Neural Network layer."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        std = float(backend.sqrt(1.0 / hidden_size))
        self.Wx = self.add_weight("Wx", shape=(input_size, hidden_size), initializer=lambda s: backend.random.randn(*s) * std)
        self.Wh = self.add_weight("Wh", shape=(hidden_size, hidden_size), initializer=lambda s: backend.random.randn(*s) * std)
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
        self.W = self.add_weight("W", shape=(hidden_size + input_size, 4 * hidden_size), initializer=lambda s: backend.random.randn(*s) * std)
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
        self.Wr = self.add_weight("Wr", shape=(hidden_size + input_size, hidden_size), initializer=lambda s: backend.random.randn(*s) * std)
        self.Wz = self.add_weight("Wz", shape=(hidden_size + input_size, hidden_size), initializer=lambda s: backend.random.randn(*s) * std)
        self.Wh = self.add_weight("Wh", shape=(hidden_size + input_size, hidden_size), initializer=lambda s: backend.random.randn(*s) * std)
        
        self.br = self.add_weight("br", shape=(hidden_size,), initializer="zeros")
        self.bz = self.add_weight("bz", shape=(hidden_size,), initializer="zeros")
        self.bh = self.add_weight("bh", shape=(hidden_size,), initializer="zeros")

    def forward(self, x: Tensor, h0: Tensor = None) -> Tensor:
        return ops.gru_cell(x, self.Wr, self.Wz, self.Wh, self.br, self.bz, self.bh, h0=h0)
