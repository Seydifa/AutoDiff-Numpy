# dnp/ops/vjp_rules.py
# Backward-compatibility shim — all logic lives in dnp/core/vjp_rules.

from dnp.core.vjp_rules import *  # noqa: F401, F403
from dnp.core.vjp_rules import (
    # Registry + decorator
    VJP_RULES,
    vjp_rule,
    # Utility helpers exposed as public API
    EPSILON,
    unbroadcast,
    # Activation forward functions (also serve as VJP registry keys)
    sigmoid,
    relu,
    leaky_relu,
    elu,
    softplus,
    swish,
    gelu,
    softmax,
    # Convolution forward functions
    conv2d,
    rot180,
    conv2d_full,
    # Internal variadic wrappers (needed by layers importing from this shim)
    _concatenate,
    _stack,
)
