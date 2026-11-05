"""Compatibility import for the shared genbonded parameter database."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("tmol_core.database.genbonded")
