# dnp/ops/vjp_rules.py
# Backward-compatibility shim — all logic has moved to dnp/core.

# Local imports
from dnp.core.vjp_rules import *  # noqa: F401, F403
from dnp.core.vjp_rules import (
    VJP_RULES,
    EPSILON,
    unbroadcast,
    sigmoid,
    relu,
    leaky_relu,
    elu,
    softplus,
    swish,
    gelu,
    softmax,
    conv2d,
    rot180,
    conv2d_full,
)
