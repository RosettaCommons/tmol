"""Compatibility import for shared ideal coords."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("tmol_core.chemical.ideal_coords")
