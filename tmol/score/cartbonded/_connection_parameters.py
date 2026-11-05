"""Compatibility alias for shared explicit connection parameter validation."""

import sys
from tmol_core.scoring import cartbonded_connections as _shared

sys.modules[__name__] = _shared
