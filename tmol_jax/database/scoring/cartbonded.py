"""Compatibility import for the shared cartbonded parameter database."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("tmol_core.database.cartbonded")
