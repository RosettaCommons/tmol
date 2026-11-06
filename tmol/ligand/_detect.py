"""Torch compatibility API over shared ligand preparation."""

from tmol_core.ligand import _detect as _core
from ._execution import _with_torch_execution


def __getattr__(name):
    return getattr(_core, name)


nonstandard_residue_info_from_smiles_via_mol2 = _with_torch_execution(
    _core.nonstandard_residue_info_from_smiles_via_mol2
)
