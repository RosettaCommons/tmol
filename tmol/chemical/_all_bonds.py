"""Compatibility import for shared all bonds."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("tmol_core.chemical.all_bonds")
