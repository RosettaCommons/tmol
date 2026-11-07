"""Map molecular Biotite identities through common canonical chemistry."""

import biotite.structure as struc
import numpy as np

from .canonical_ordering import (
    _canonical_form_from_records,
    default_canonical_ordering,
    default_packed_block_types,
)
from .pdb_parsing import _records_from_biotite
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
    from tmol_core.io.atom_array import (
        with_metal_coordination_typed,
        covalent_bonds_from_biotite,
        metal_coordination_from_biotite,
    )
    from tmol_core.io.polymer_gaps import break_polymer_gaps, orient_polymer_gap_flags

    co = canonical_ordering or default_canonical_ordering()
    if prepare_input:
        return _prepared_canonical_form(
            atoms,
            co,
            missing_density_distance_threshold,
            filter_missing_mainchain,
            cut_partners,
        )
    atoms = with_metal_coordination_typed(atoms)
    models = list(atoms) if isinstance(atoms, struc.AtomArrayStack) else [atoms]
    if not models:
        raise ValueError("structure contains no models")
    mapped = [
        _canonical_form_from_records(
            _records_from_biotite(model),
            co,
            missing_density_distance_threshold=0,
            return_atom_mapping=True,
        )
        for model in models
    ]
    forms = [form for form, _ in mapped]
    combined = {
        key: np.concatenate([form[key] for form in forms], axis=0) for key in forms[0]
    }
    residues, valid, slots = mapped[0][1]
    from tmol_core.io.atom_metadata import populate_optional_atom_metadata
    from tmol_core.io import structure_context as metadata_policy

    b_factors, occupancy = populate_optional_atom_metadata(
        atoms,
        len(models),
        len(combined["res_types"][0]),
        co.max_n_canonical_atoms,
        residues[valid],
        slots,
        valid,
    )
    combined["atom_b_factor"] = b_factors
    combined["atom_occupancy"] = occupancy
    for key, source in (
        ("metal_origins", metadata_policy._metal_origins(atoms)),
        ("residue_annotations", metadata_policy._residue_annotations(atoms)),
    ):
        if source is not None:
            combined[key] = np.broadcast_to(source, (len(models), *source.shape)).copy()
    types, chains = combined["res_types"][0], combined["chain_id"][0]
    bonds, disulfides, inferred = covalent_bonds_from_biotite(
        models[0],
        co,
        residues,
        types,
        valid,
        slots,
    )
    gaps = np.zeros((len(types), 2), bool)
    bonds = break_polymer_gaps(
        gaps,
        bonds,
        inferred,
        combined["coords"],
        types,
        chains,
        co,
        missing_density_distance_threshold,
    )
    orient_polymer_gap_flags(gaps, chains, types, bonds, co)
    combined["res_gap"] = np.broadcast_to(gaps, (len(models), *gaps.shape)).copy()

    def pose_rows(rows):
        return np.concatenate(
            [
                np.column_stack((np.full(len(rows), p, np.int64), rows))
                for p in range(len(models))
            ]
        )

    combined["covalent_bonds"] = pose_rows(bonds)
    if disulfides is not None:
        combined["disulfides"] = pose_rows(disulfides)
    combined["metal_coordination"] = pose_rows(
        metal_coordination_from_biotite(models[0], residues, valid, slots)
    )
    return combined


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
