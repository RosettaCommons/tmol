"""Torch compatibility API over shared ligand preparation."""

from tmol_core.ligand import _openbabel_compat as _core
from ._execution import _with_torch_execution


def __getattr__(name):
    return getattr(_core, name)


_build_charged_3d_mol2_mol = _with_torch_execution(_core._build_charged_3d_mol2_mol)
