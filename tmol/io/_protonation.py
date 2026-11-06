"""Compatibility alias for common structure-input chemistry."""

import sys
from tmol_core.io import protonation_input as _implementation

sys.modules[__name__] = _implementation
