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
    initial_gaps = None
    if prepare_input:
        from tmol_core.io import structure_context as common

        atoms = common._normalize_input_identifiers(atoms, co.name3_aliases)
        atoms = common._with_input_chemistry_normalized(
            atoms, common._metal_atom_names(co=co)
        )
        atoms, initial_gaps = common._filter_supported_atoms_and_connectivity(
            atoms,
            co,
            cut_partners=cut_partners,
            filter_missing_mainchain=filter_missing_mainchain,
        )
    atoms = with_metal_coordination_typed(atoms)
    models = list(atoms) if isinstance(atoms, struc.AtomArrayStack) else [atoms]
    if not models:
        raise ValueError("structure contains no models")
    prepared_atom_indices = None
    if prepare_input:
        from tmol_core.io.canonical_atom_mapping import (
            map_atoms_to_canonical,
            map_neutral_terminal_hydrogens,
        )
        from tmol_core.io.protonation_input import PROTONATION_VARIANT

        template = models[0]
        residue_indices = struc.get_all_residue_positions(template)
        valid, indices, _ = map_atoms_to_canonical(
            co,
            residue_indices,
            template.res_name,
            template.atom_name,
            template.element,
        )
        if PROTONATION_VARIANT in template.get_annotation_categories():
            indices = map_neutral_terminal_hydrogens(
                co,
                residue_indices,
                template.res_name,
                template.atom_name,
                template.get_annotation(PROTONATION_VARIANT),
                valid,
                indices,
            )
        prepared_atom_indices = np.full(len(template), -1, np.int64)
        prepared_atom_indices[valid] = indices
    mapped = [
        _canonical_form_from_records(
            _records_from_biotite(model),
            co,
            missing_density_distance_threshold=0,
            return_atom_mapping=True,
            prepared_atom_indices=prepared_atom_indices,
        )
        for model in models
    ]
    forms = [form for form, _ in mapped]
    combined = {
        key: np.concatenate([form[key] for form in forms], axis=0) for key in forms[0]
    }
    residues, valid, slots = mapped[0][1]
    if prepare_input:
        chain_ids, chain_labels, insertion_codes, labels, _ = (
            common._extract_residue_metadata(atoms, initial_gaps)
        )
        count = len(models)
        for name, values in (
            ("chain_id", chain_ids),
            ("chain_labels", chain_labels),
            ("res_ins_codes", insertion_codes),
            ("res_num", labels),
        ):
            combined[name] = np.broadcast_to(values, (count, len(values))).copy()
        from tmol_core.io.protonation_input import PROTONATION_VARIANT

        if PROTONATION_VARIANT in atoms.get_annotation_categories():
            values = atoms.get_annotation(PROTONATION_VARIANT)[
                struc.get_residue_starts(atoms)
            ]
            combined["protonation_variants"] = np.broadcast_to(
                values, (count, len(values))
            ).copy()
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
    gaps = np.zeros((len(types), 2), bool) if initial_gaps is None else initial_gaps
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
