"""Compatibility import for common Lewis bond-order preparation."""

import sys
from tmol_core.chemical import bond_orders as _implementation

sys.modules[__name__] = _implementation
