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
