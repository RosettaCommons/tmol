"""Shared ligand chemistry with explicit parameter and execution services."""

from importlib import import_module

_PUBLIC_MODULES = {
    "ConformerExecution": "execution",
    "FRAGMENT_ID_ANNOTATION": "_fragmentation",
    "LigandPreparation": "_registry",
    "LigandPreparationError": "_preparation",
    "NonStandardResidueInfo": "_detect",
    "TMOL_FORMAT_VERSION": "_params_file",
    "inject_params_file": "_params_file",
    "inject_params_files": "_params_file",
    "load_params_file": "_params_file",
    "write_params_file": "_params_io",
    "write_params_from_mol2": "_params_io",
    "ligand_smiles_from_atom_array": "_structure_to_smiles",
    "nonstandard_residue_info_from_mol2": "_detect",
    "nonstandard_residue_info_from_smiles_via_mol2": "_detect",
    "prepare_ligand_from_cif": "_preparation",
    "prepare_ligand_from_mol2": "_preparation",
    "prepare_ligand_from_smiles": "_preparation",
    "prepare_ligands": "_preparation",
    "prepare_single_ligand": "_preparation",
    "smiles_from_block_type": "_block_type_smiles",
}
__all__ = list(_PUBLIC_MODULES)


def __getattr__(name):
    module = _PUBLIC_MODULES.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
