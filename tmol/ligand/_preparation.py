"""Torch compatibility API over shared ligand preparation."""

from tmol_core.ligand import _preparation as _core
from ._execution import _with_torch_execution


def __getattr__(name):
    return getattr(_core, name)


_prepare_polymer_residue = _with_torch_execution(_core._prepare_polymer_residue)
_prepare_ligand_via_smiles = _with_torch_execution(_core._prepare_ligand_via_smiles)
_state_variants = _with_torch_execution(_core._state_variants)
prepare_ligands = _with_torch_execution(_core.prepare_ligands)
_ligand_info_from_cif = _with_torch_execution(_core._ligand_info_from_cif)
_inject_single = _with_torch_execution(_core._inject_single)
prepare_ligand_from_cif = _with_torch_execution(_core.prepare_ligand_from_cif)
prepare_ligand_from_smiles = _with_torch_execution(_core.prepare_ligand_from_smiles)
_prepare_ligands_from_smiles = _with_torch_execution(_core._prepare_ligands_from_smiles)
_prepare_mol2 = _with_torch_execution(_core._prepare_mol2)
prepare_ligand_from_mol2 = _with_torch_execution(_core.prepare_ligand_from_mol2)
