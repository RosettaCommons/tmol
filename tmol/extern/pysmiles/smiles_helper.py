"""Compatibility import for :mod:`tmol_core.extern.pysmiles.smiles_helper`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("tmol_core.extern.pysmiles.smiles_helper")
