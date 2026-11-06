"""Map molecular Biotite identities through common canonical chemistry."""

import biotite.structure as struc
import numpy as np

from .canonical_ordering import (
    _canonical_form_from_records,
    _polymer_gaps,
    default_canonical_ordering,
    default_packed_block_types,
)
from .pdb_parsing import _records_from_biotite
from .pose_stack_construction import pose_stack_from_canonical_form


def canonical_form_from_biotite(
    atoms, *, canonical_ordering=None, missing_density_distance_threshold=2.4
):
    """Convert an AtomArray or same-topology AtomArrayStack to canonical form."""
    if atoms.bonds is not None and atoms.bonds.get_bond_count():
        raise NotImplementedError(
            "explicit Biotite bonds require the canonical covalent_bonds, "
            "disulfides and metal_coordination adapters"
        )
    co = canonical_ordering or default_canonical_ordering()
    models = list(atoms) if isinstance(atoms, struc.AtomArrayStack) else [atoms]
    if not models:
        raise ValueError("structure contains no models")
    forms = [
        _canonical_form_from_records(
            _records_from_biotite(model), co, missing_density_distance_threshold=0
        )
        for model in models
    ]
    combined = {
        key: np.concatenate([form[key] for form in forms], axis=0) for key in forms[0]
    }
    combined["res_gap"] = _polymer_gaps(
        co,
        combined["res_types"],
        combined["chain_id"],
        combined["coords"],
        missing_density_distance_threshold,
    )
    return combined


def pose_stack_from_biotite(
    atoms,
    *,
    canonical_ordering=None,
    packed_block_types=None,
    missing_density_distance_threshold=2.4,
    **kwargs,
):
    """Reconstruct molecular input using its supplied ordering and palette."""
    co = canonical_ordering or default_canonical_ordering()
    canonical = canonical_form_from_biotite(
        atoms,
        canonical_ordering=co,
        missing_density_distance_threshold=missing_density_distance_threshold,
    )
    canonical["canonical_ordering"] = co
    return pose_stack_from_canonical_form(
        canonical, packed_block_types or default_packed_block_types(), **kwargs
    )
