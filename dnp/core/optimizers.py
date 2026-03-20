"""
dnp/core/optimizers.py
=======================
Optimization algorithms for training neural networks.

This module provides a family of optimizers for gradient-based learning:
  - Optimizer (base class)
  - SGD (Stochastic Gradient Descent)
  - Momentum (SGD with momentum)
  - RMSprop (Root Mean Square Propagation)
  - Adagrad (Adaptive Gradient)
  - Adam (Adaptive Moment Estimation)
  - AdamW (Adam with Weight Decay)
"""

# Standard library
from typing import List, Dict, Any

# Local imports
from .backend import get_xp


class Optimizer:
    """
    Base class for all optimization algorithms.

    An optimizer takes a set of learnable parameters and a learning rate,
    then provides methods to zero gradients and perform optimization steps.

    Parameters
    ----------
    parameters : iterable
        Iterable of parameters (tensors) to optimize.
    lr : float, default=0.01
        Learning rate controlling the step size in the direction of
        negative gradient.

    Attributes
    ----------
    parameters : list
        List of parameters managed by this optimizer.
    lr : float
        Learning rate.

    Examples
    --------
    >>> from dnp.core import Optimizer
    >>> class MyOptimizer(Optimizer):
    ...     def step(self):
    ...         for p in self.parameters:
    ...             if p.grad is not None:
    ...                 p[...] = p - self.lr * p.grad
    """

    def __init__(self, parameters: List[Any], lr: float = 0.01):
        """Initialize the optimizer with parameters and learning rate."""
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """
        Zero out all parameter gradients.

        This function must be called before each backward pass to prevent
        accumulation of gradients from previous iterations.
        """
        for p in self.parameters:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.fill(0.0)

    def step(self) -> None:
        """
        Perform a single optimization step.

        This method updates the parameters based on their gradients.
        Must be implemented by subclasses.

        Raises
        ------
        NotImplementedError
            This method must be implemented by optimizer subclasses.
        """
        raise NotImplementedError("Subclasses must implement step()")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Implements the vanilla SGD algorithm with a constant learning rate:

    .. math::
        w_{t+1} = w_t - \\alpha \\nabla L(w_t)

    where :math:`\\alpha` is the learning rate and :math:`\\nabla L(w_t)`
    is the gradient of the loss at time t.

    Parameters
    ----------
    parameters : iterable
        Iterable of parameters (tensors) to optimize.
    lr : float, default=0.01
        Learning rate for the optimization step.

    Attributes
    ----------
    parameters : list
        List of parameters managed by this optimizer.
    lr : float
        Learning rate.

    Examples
    --------
    >>> from dnp.core import SGD
    >>> optimizer = SGD(model.parameters(), lr=0.01)
    >>> for epoch in range(num_epochs):
    ...     output = model(x)
    ...     loss = criterion(output, y)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    """

    def step(self) -> None:
        """
        Perform a single SGD step.

        Updates each parameter by subtracting the learning rate times
        its gradient:

        .. math::
            p \\leftarrow p - \\alpha \\nabla L(p)
        """
        for p in self.parameters:
            if p.grad is not None:
                # W = W - lr * dL/dW
                p[...] = p - self.lr * p.grad


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.
    
    Adam is an adaptive learning rate optimizer that maintains running
    estimates of both first-order moments (mean) and second-order moments
    (variance) of the gradients. It tends to work well across a wide
    range of problems and is less sensitive to learning rate selection.
    
    The update rule is:
    
    .. math::
        m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\\\
        v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
        \\hat{m}_t &= m_t / (1 - \\beta_1^t) \\\\
        \\hat{v}_t &= v_t / (1 - \\beta_2^t) \\\\
        w_{t+1} &= w_t - \\alpha \\hat{m}_t / (\\sqrt{\\hat{v}_t} + \\epsilon)
    
    where :math:`g_t` is the gradient at time t, :math:`\\beta_1` and
    :math:`\\beta_2` are exponential decay rates, and :math:`\\epsilon`
    is a small constant for numerical stability.
    
    Parameters
    ----------
    parameters : iterable
        Iterable of parameters (tensors) to optimize.
    lr : float, default=0.001
        Learning rate (often called alpha in the literature).
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates.
    epsilon : float, default=1e-8
        Small constant for numerical stability and to prevent division by zero.
    
    Attributes
    ----------
    parameters : list
        List of parameters managed by this optimizer.
    lr : float
        Learning rate.
    beta1 : float
        Decay rate for first moment.
    beta2 : float
        Decay rate for second moment.
    epsilon : float
        Numerical stability constant.
    t : int
        Number of optimization steps performed (for bias correction).
    m : dict
        First moment estimates (moving averages of gradients).
    v : dict
        Second moment estimates (moving averages of gradient squares).
    
    References
    ----------
    Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
    arXiv preprint arXiv:1412.6980.
    
    Examples
    --------
    >>> from dnp.core import Adam
    >>> optimizer = Adam(model.parameters(), lr=0.001)
    >>> for epoch in range(num_epochs):
    ...     output = model(x)
    ...     loss = criterion(output, y)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        parameters: List[Any],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """
        Initialize the Adam optimizer.

        Parameters
        ----------
        parameters : iterable
            Iterable of parameter tensors to optimize.
        lr : float, default=0.001
            Learning rate.
        beta1 : float, default=0.9
            Exponential decay rate for first moment estimates.
        beta2 : float, default=0.999
            Exponential decay rate for second moment estimates.
        epsilon : float, default=1e-8
            Small constant for numerical stability.
        """
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step counter

        # Initialize first and second moment estimates for each parameter
        # We use the Python object ID as the dictionary key to uniquely
        # identify each parameter tensor.
        # State is allocated on the same device as the parameter.
        self.m: Dict[int, Any] = {
            id(p): get_xp(p.data).zeros_like(p.data) for p in self.parameters
        }
        self.v: Dict[int, Any] = {
            id(p): get_xp(p.data).zeros_like(p.data) for p in self.parameters
        }

    def step(self) -> None:
        """
        Perform a single Adam optimization step.

        Update each parameter using adaptive learning rates based on
        estimates of first and second moments of the gradients with
        bias correction.
        """
        self.t += 1

        for p in self.parameters:
            if p.grad is None:
                continue

            pid = id(p)
            grad = p.grad

            # 1. Update biased first moment estimate (exponential moving average)
            #    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            self.m[pid] = self.beta1 * self.m[pid] + (1.0 - self.beta1) * grad

            # 2. Update biased second moment estimate
            #    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            self.v[pid] = self.beta2 * self.v[pid] + (1.0 - self.beta2) * (grad**2)

            # 3. Compute bias-corrected first moment estimate
            #    m_hat = m_t / (1 - beta1^t)
            # This is crucial: without bias correction, the early steps
            # would be dominated by the initialization (zeros), leading
            # to slow convergence
            m_hat = self.m[pid] / (1.0 - self.beta1**self.t)

            # 4. Compute bias-corrected second moment estimate
            #    v_hat = v_t / (1 - beta2^t)
            v_hat = self.v[pid] / (1.0 - self.beta2**self.t)

            # 5. Update parameters
            #    p = p - lr * m_hat / (sqrt(v_hat) + epsilon)
            # The denominator sqrt(v_hat) + epsilon provides adaptive per-parameter
            # learning rates, reducing the effective learning rate for parameters
            # with large gradient variance and increasing it for those with small variance
            xp = get_xp(v_hat)
            p[...] = p - self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)


class Momentum(Optimizer):
    """
    SGD with Momentum optimizer.

    Momentum accelerates gradient descent by accumulating a velocity vector
    in the direction of the gradient. This helps escape local minima and
    smooths out oscillations in noisy gradients.

    The update rule is:

    .. math::
        v_t &= \\beta v_{t-1} + g_t \\\\
        w_{t+1} &= w_t - \\alpha v_t

    where :math:`v_t` is the velocity vector and :math:`\\beta` is the
    momentum coefficient.

    Parameters
    ----------
    parameters : iterable
        Iterable of parameters (tensors) to optimize.
    lr : float, default=0.01
        Learning rate.
    momentum : float, default=0.9
        Momentum coefficient. Higher values accumulate more history.
    dampening : float, default=0.0
        Dampening factor for momentum. Set to 0 for standard momentum.
    nesterov : bool, default=False
        Whether to use Nesterov momentum (look-ahead variant).

    Attributes
    ----------
    parameters : list
        List of parameters managed by this optimizer.
    lr : float
        Learning rate.
    momentum : float
        Momentum coefficient.
    dampening : float
        Dampening for momentum.
    nesterov : bool
        Use Nesterov momentum flag.
    velocity : dict
        Accumulated velocities for each parameter.

    Examples
    --------
    >>> from dnp.core import Momentum
    >>> optimizer = Momentum(model.parameters(), lr=0.01, momentum=0.9)
    >>> for epoch in range(num_epochs):
    ...     output = model(x)
    ...     loss = criterion(output, y)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        parameters: List[Any],
        lr: float = 0.01,
        momentum: float = 0.9,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        """Initialize the Momentum optimizer."""
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.velocity: Dict[int, Any] = {
            id(p): get_xp(p.data).zeros_like(p.data) for p in self.parameters
        }

    def step(self) -> None:
        """
        Perform a single momentum step.

        Accumulates velocity in the gradient direction and uses it for
        parameter updates, optionally with Nesterov acceleration.
        """
        for p in self.parameters:
            if p.grad is None:
                continue

            pid = id(p)
            grad = p.grad

            # Accumulate velocity: v = momentum * v + (1 - dampening) * grad
            buf = self.velocity[pid]
            buf[:] = self.momentum * buf + (1.0 - self.dampening) * grad

            # Update parameters
            if self.nesterov:
                # Nesterov momentum: look ahead by adding momentum term
                # p = p - lr * (grad + momentum * v)
                p[...] = p - self.lr * (grad + self.momentum * buf)
            else:
                # Standard momentum: p = p - lr * v
                p[...] = p - self.lr * buf


class RMSprop(Optimizer):
    """
    Root Mean Square Propagation (RMSprop) optimizer.

    RMSprop divides the learning rate by an exponentially decaying average
    of squared gradients. This provides an adaptive per-parameter learning
    rate that works well with non-stationary objectives.

    The update rule is:

    .. math::
        v_t &= \\alpha v_{t-1} + (1 - \\alpha) g_t^2 \\\\
        w_{t+1} &= w_t - \\frac{\\eta}{\\sqrt{v_t} + \\epsilon} g_t

    where :math:`v_t` is the exponentially decaying average of squared
    gradients and :math:`\\alpha` is the decay rate.

    Parameters
    ----------
    parameters : iterable
        Iterable of parameters (tensors) to optimize.
    lr : float, default=0.001
        Learning rate.
    alpha : float, default=0.99
        Decay rate for the moving average of squared gradients.
    epsilon : float, default=1e-8
        Small constant for numerical stability.
    centered : bool, default=False
        If True, use centered RMSprop by normalizing by the centered variance.

    Attributes
    ----------
    parameters : list
        List of parameters managed by this optimizer.
    lr : float
        Learning rate.
    alpha : float
        Decay rate.
    epsilon : float
        Numerical stability constant.
    centered : bool
        Use centered RMSprop flag.
    sq_avg : dict
        Exponential moving average of squared gradients.
    buffer : dict
        Buffer for centered RMSprop (if enabled).

    Examples
    --------
    >>> from dnp.core import RMSprop
    >>> optimizer = RMSprop(model.parameters(), lr=0.001, alpha=0.99)
    >>> for epoch in range(num_epochs):
    ...     output = model(x)
    ...     loss = criterion(output, y)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        parameters: List[Any],
        lr: float = 0.001,
        alpha: float = 0.99,
        epsilon: float = 1e-8,
        centered: bool = False,
    ):
        """Initialize the RMSprop optimizer."""
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.epsilon = epsilon
        self.centered = centered

        self.sq_avg: Dict[int, Any] = {
            id(p): get_xp(p.data).zeros_like(p.data) for p in self.parameters
        }
        if centered:
            self.buffer: Dict[int, Any] = {
                id(p): get_xp(p.data).zeros_like(p.data) for p in self.parameters
            }

    def step(self) -> None:
        """
        Perform a single RMSprop step.

        Update each parameter using adaptive learning rates based on
        exponentially decaying averages of squared gradients.
        """
        for p in self.parameters:
            if p.grad is None:
                continue

            pid = id(p)
            grad = p.grad

            # Update square gradient average: v = alpha * v + (1-alpha) * g^2
            self.sq_avg[pid] = self.alpha * self.sq_avg[pid] + (1.0 - self.alpha) * (
                grad**2
            )

            if self.centered:
                # Centered RMSprop: normalize by centered variance
                self.buffer[pid] = (
                    self.alpha * self.buffer[pid] + (1.0 - self.alpha) * grad
                )
                denominator = self.sq_avg[pid] - self.buffer[pid] ** 2 + self.epsilon
            else:
                denominator = self.sq_avg[pid] + self.epsilon

            # Update parameter: p = p - lr * grad / sqrt(denominator)
            xp = get_xp(denominator)
            p[...] = p - self.lr * grad / xp.sqrt(denominator)


class Adagrad(Optimizer):
    """
    Adaptive Gradient (Adagrad) optimizer.

    Adagrad adapts the learning rate based on historical gradients.
    Parameters that receive large gradients have their learning rate
    reduced, while parameters that receive small updates have their
    learning rate increased. This is effective for sparse data.

    The update rule is:

    .. math::
        G_t &= G_{t-1} + g_t^2 \\\\
        w_{t+1} &= w_t - \\frac{\\eta}{\\sqrt{G_t} + \\epsilon} g_t

    where :math:`G_t` is the cumulative sum of squared gradients.

    Parameters
    ----------
    parameters : iterable
        Iterable of parameters (tensors) to optimize.
    lr : float, default=0.01
        Learning rate.
    epsilon : float, default=1e-8
        Small constant for numerical stability.

    Attributes
    ----------
    parameters : list
        List of parameters managed by this optimizer.
    lr : float
        Learning rate.
    epsilon : float
        Numerical stability constant.
    sq_sum : dict
        Cumulative sum of squared gradients.

    Examples
    --------
    >>> from dnp.core import Adagrad
    >>> optimizer = Adagrad(model.parameters(), lr=0.01)
    >>> for epoch in range(num_epochs):
    ...     output = model(x)
    ...     loss = criterion(output, y)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()

    Notes
    -----
    Adagrad accumulates historical gradients without decay, which can
    cause the learning rate to decrease too quickly in long training runs.
    RMSprop or Adam are often preferred for this reason.
    """

    def __init__(
        self,
        parameters: List[Any],
        lr: float = 0.01,
        epsilon: float = 1e-8,
    ):
        """Initialize the Adagrad optimizer."""
        super().__init__(parameters, lr)
        self.epsilon = epsilon
        self.sq_sum: Dict[int, Any] = {
            id(p): get_xp(p.data).zeros_like(p.data) for p in self.parameters
        }

    def step(self) -> None:
        """
        Perform a single Adagrad step.

        Accumulates squared gradients and uses their square root to
        scale the learning rate for each parameter individually.
        """
        for p in self.parameters:
            if p.grad is None:
                continue

            pid = id(p)
            grad = p.grad

            # Accumulate squared gradients: G = G_prev + g^2
            self.sq_sum[pid] = self.sq_sum[pid] + grad**2

            # Update parameter: p = p - lr * grad / sqrt(G + epsilon)
            xp = get_xp(self.sq_sum[pid])
            p[...] = p - self.lr * grad / (xp.sqrt(self.sq_sum[pid]) + self.epsilon)


class AdamW(Optimizer):
    """
    AdamW (Adam with decoupled Weight Decay) optimizer.

    AdamW is a variant of Adam that decouples weight decay from the
    gradient-based update. This provides more principled L2 regularization
    compared to L2 penalty added to the loss (which is what Adam with
    weight decay does).

    The update rule is:

    .. math::
        m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\\\
        v_t &= \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
        \\hat{m}_t &= m_t / (1 - \\beta_1^t) \\\\
        \\hat{v}_t &= v_t / (1 - \\beta_2^t) \\\\
        w_{t+1} &= w_t - \\alpha \\hat{m}_t / (\\sqrt{\\hat{v}_t} + \\epsilon) - \\lambda w_t

    where the last term is decoupled weight decay (not scaled by learning rate).

    Parameters
    ----------
    parameters : iterable
        Iterable of parameters (tensors) to optimize.
    lr : float, default=0.001
        Learning rate.
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates.
    epsilon : float, default=1e-8
        Small constant for numerical stability.
    weight_decay : float, default=0.01
        Weight decay coefficient (L2 regularization strength).

    Attributes
    ----------
    parameters : list
        List of parameters managed by this optimizer.
    lr : float
        Learning rate.
    beta1 : float
        Decay rate for first moment.
    beta2 : float
        Decay rate for second moment.
    epsilon : float
        Numerical stability constant.
    weight_decay : float
        Weight decay coefficient.
    t : int
        Number of optimization steps performed.
    m : dict
        First moment estimates.
    v : dict
        Second moment estimates.

    References
    ----------
    Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay
    regularization. In International Conference on Learning Representations.

    Examples
    --------
    >>> from dnp.core import AdamW
    >>> optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    >>> for epoch in range(num_epochs):
    ...     output = model(x)
    ...     loss = criterion(output, y)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        parameters: List[Any],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """Initialize the AdamW optimizer."""
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0

        self.m: Dict[int, Any] = {
            id(p): get_xp(p.data).zeros_like(p.data) for p in self.parameters
        }
        self.v: Dict[int, Any] = {
            id(p): get_xp(p.data).zeros_like(p.data) for p in self.parameters
        }

    def step(self) -> None:
        """
        Perform a single AdamW step.

        Update parameters using adaptive learning rates with decoupled
        weight decay regularization.
        """
        self.t += 1

        for p in self.parameters:
            if p.grad is None:
                continue

            pid = id(p)
            grad = p.grad

            # Update biased first moment estimate
            self.m[pid] = self.beta1 * self.m[pid] + (1.0 - self.beta1) * grad

            # Update biased second moment estimate
            self.v[pid] = self.beta2 * self.v[pid] + (1.0 - self.beta2) * (grad**2)

            # Bias correction
            m_hat = self.m[pid] / (1.0 - self.beta1**self.t)
            v_hat = self.v[pid] / (1.0 - self.beta2**self.t)

            # AdamW update: adaptive step + decoupled weight decay
            # The key difference from Adam: weight decay is applied directly,
            # not scaled by the adaptive learning rate
            xp = get_xp(v_hat)
            adaptive_update = self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)
            weight_decay_update = self.weight_decay * self.lr * p.data
            p[...] = p.data - adaptive_update - weight_decay_update
