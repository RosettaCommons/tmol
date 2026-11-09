"""Compatibility import for the shared scientific implementation."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("tmol_core.scoring.hash_util")
