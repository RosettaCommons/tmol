"""Compatibility import for shared ligand chemistry."""

import sys
from tmol_core.ligand import _params_file as _implementation

sys.modules[__name__] = _implementation
