"""Map molecular Biotite identities through common canonical chemistry."""

import numpy as np

from .canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
)
from tmol_core.io.canonical_records import (
    canonical_form_from_biotite as _canonical_form_from_biotite,
)
from .pose_stack_construction import pose_stack_from_canonical_form


def canonical_form_from_biotite(
    atoms,
    *,
    canonical_ordering=None,
    missing_density_distance_threshold=2.4,
    prepare_input=False,
    cut_partners=frozenset(),
    filter_missing_mainchain=True,
):
    """Convert an AtomArray or same-topology AtomArrayStack to canonical form."""
    co = canonical_ordering or default_canonical_ordering()
    if prepare_input:
        return _prepared_canonical_form(
            atoms,
            co,
            missing_density_distance_threshold,
            filter_missing_mainchain,
            cut_partners,
        )
    return _canonical_form_from_biotite(
        atoms, co, missing_density_distance_threshold=missing_density_distance_threshold
    )


def _prepared_canonical_form(atoms, co, distance, filter_mainchain, cut_partners):
    """Expose the common complete molecular input plan with JAX field aliases."""
    from tmol_core.io.canonical_atom_array import prepare_canonical_atom_array

    plan = prepare_canonical_atom_array(
        atoms,
        co,
        missing_density_distance_threshold=distance,
        filter_missing_mainchain=filter_mainchain,
        cut_partners=cut_partners,
    )
    return _canonical_form_from_plan(plan)


def _canonical_form_from_plan(plan):
    """Apply frontend field aliases without repeating chemical preparation."""
    aliases = {
        "res_labels": "res_num",
        "residue_insertion_codes": "res_ins_codes",
        "res_not_connected": "res_gap",
    }
    fields = {
        aliases.get(name, name): value
        for name, value in plan.fields.items()
        if value is not None
    }
    fields["res_name"] = np.broadcast_to(
        np.asarray(plan.residue_names, dtype=object), fields["res_types"].shape
    ).copy()
    fields["atom_is_present"] = np.isfinite(fields["coords"]).all(axis=-1)
    return fields


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
