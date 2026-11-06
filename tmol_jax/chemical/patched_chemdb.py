"""Shared residue patching followed by JAX host-topology refinement."""

import attr
from tmol_core.chemical.patching import (
    PatchedChemicalDatabase as _PatchedChemicalDatabase,
)
from tmol_core.chemical.patching import (  # noqa: F401
    RestypeGraphBuilder,
    _icoors_in_dependency_order,
    _only_adds_connections,
    _patch_preserves_torsion_support,
    _validate_patch_atom_aliases,
    _validate_patch_atom_references,
    _validate_patch_bonds,
    _validate_patch_fields,
    _validate_patch_icoors,
    _validate_patch_scope,
    _validate_raw_residue_atoms,
    _validate_raw_residue_bonds,
    _validate_raw_residue_connections,
    _validate_raw_residue_icoors,
    _validate_raw_residue_metal_sites,
    _validate_raw_residue_torsions,
    _assert_no_orphaned_icoors as assert_no_orphaned_icoors,
    _do_patch as do_patch,
    _get_modified_atoms as get_modified_atoms,
    _modify_atom as modify_atom,
    _patch_residue as patch_residue,
    _pattern_namemaps as pattern_namemaps,
    _remove_atom as remove_atom,
    _update_icoor as update_icoor,
    _validate_patch as validate_patch,
    _validate_raw_residue as validate_raw_residue,
)
from tmol_jax.chemical.restypes import _refined_from_raw


class PatchedChemicalDatabase(_PatchedChemicalDatabase):
    """Patched records materialized as JAX host residue types."""

    @classmethod
    def from_chem_db(cls, chemdb):
        patched = _PatchedChemicalDatabase.from_chem_db(chemdb)
        fields = attr.asdict(patched, recurse=False)
        fields["residues"] = tuple(_refined_from_raw(r) for r in patched.residues)
        return cls(**fields)


def default_patched_chemdb():
    from tmol_jax.database import ParameterDatabase

    return ParameterDatabase.get_default().chemical
