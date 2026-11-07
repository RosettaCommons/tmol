"""Compatibility import for the shared molecular sequence grammar."""

import sys
from tmol_core.pose import sequence as _implementation

sys.modules[__name__] = _implementation
