"""Torch compatibility API over shared ligand preparation."""

from tmol_core.ligand import _params_io as _core
from ._execution import _with_torch_execution


def __getattr__(name):
    return getattr(_core, name)


write_params_from_mol2 = _with_torch_execution(_core.write_params_from_mol2)
