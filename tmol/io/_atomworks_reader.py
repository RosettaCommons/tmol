"""Compatibility import for the shared structure chemistry implementation."""

import sys
from tmol_core.io import atomworks_reader as _shared

sys.modules[__name__] = _shared
