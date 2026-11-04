"""Compatibility import for :mod:`tmol_core._yaml`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("tmol_core._yaml")
