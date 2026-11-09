"""Compatibility import for shared metal coordination parameter records."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("tmol_core.database.metal_coordination")
