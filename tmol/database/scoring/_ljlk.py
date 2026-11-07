"""Compatibility alias for the shared LJ/LK database schema."""

import sys
from tmol_core.database import ljlk as _shared

sys.modules[__name__] = _shared
