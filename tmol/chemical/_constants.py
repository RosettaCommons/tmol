"""Compatibility import for shared constants."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("tmol_core.chemical.constants")
