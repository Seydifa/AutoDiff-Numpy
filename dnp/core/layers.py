"""
Backward compatibility proxy for legacy `dnp.core.layers` imports.
"""

import warnings

warnings.warn(
    "dnp.core.layers is deprecated. Use dnp.layers instead.",
    DeprecationWarning,
    stacklevel=2,
)

from dnp.layers import *
from dnp.layers import __all__
