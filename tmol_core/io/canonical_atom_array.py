"""Prepared AtomArray chemistry, canonical layout and host metadata.

Tensor frontends only place these arrays and, for Atom37, overlay differentiable
coordinates. No tensor values enter the shared topology policy.
"""

from dataclasses import dataclass

import biotite.structure as struc
import numpy as np

from . import structure_context as context
from .atom_array import (
    covalent_bonds_from_biotite,
    metal_coordination_from_biotite,
    template_array,
    with_metal_coordination_typed,
)
from .atom_metadata import populate_optional_atom_metadata
from .canonical_atom_mapping import (
    map_atoms_to_canonical,
    map_neutral_terminal_hydrogens,
)
from .input_termini import EXPLICIT_TERMINI
from .polymer_gaps import break_polymer_gaps, orient_polymer_gap_flags
from .protonation_input import PROTONATION_VARIANT


@dataclass(frozen=True)
class CanonicalAtomArrayPlan:
    structure: object
    residue_names: np.ndarray
    valid_atom_mask: np.ndarray
    valid_res_inds: np.ndarray
    valid_atom_inds: np.ndarray
    fields: dict


def bonds_for_poses(bonds, n_poses):
    """Prefix each bond row with its pose index, preserving absent disulfides."""
    if bonds is None:
        return None
    return np.column_stack(
        (np.repeat(np.arange(n_poses), len(bonds)), np.tile(bonds, (n_poses, 1)))
    ).astype(np.int64, copy=False)


def _optional_metadata(atoms, n_poses, n_res, max_atoms, valid, residues, slots):
    b_factors, occupancy = populate_optional_atom_metadata(
        atoms, n_poses, n_res, max_atoms, residues, slots, valid
    )
    result = dict(atom_b_factor=b_factors, atom_occupancy=occupancy)
    for key, values in (
        ("metal_origins", context._metal_origins(atoms)),
        ("residue_annotations", context._residue_annotations(atoms)),
    ):
        result[key] = (
            None if values is None else np.repeat(values[None], n_poses, axis=0)
        )
    result["protonation_variants"] = None
    if PROTONATION_VARIANT in atoms.get_annotation_categories():
        values = atoms.get_annotation(PROTONATION_VARIANT)[
            struc.get_residue_starts(atoms)
        ]
        result["protonation_variants"] = np.repeat(
            values[None], n_poses, axis=0
        ).astype(np.int64)
    return result


def prepare_canonical_atom_array(
    atoms,
    co,
    *,
    missing_density_distance_threshold=2.4,
    filter_missing_mainchain=True,
    cut_partners=frozenset(),
    atom37_n_poses=None,
    chemical_database=None,
):
    """Resolve prepared chemistry once for all poses with the same topology.

    ``atom37_n_poses`` requests fixed topology and metadata for an external
    coordinate tensor. It disables coordinate-based filtering and gap detection,
    and leaves ``fields['coords']`` unset (None); the frontend validates and
    overlays that tensor without detaching its gradients. Ordinary coordinates
    retain the mature float32 canonical storage contract. An explicit
    ``chemical_database`` additionally annotates complete supplied terminal
    chemistry without generating atoms or protonating incomplete input.
    """
    atoms = context._normalize_input_identifiers(atoms, co.name3_aliases)
    atoms = context._with_input_chemistry_normalized(
        atoms, context._metal_atom_names(co=co)
    )
    if chemical_database is not None:
        atoms = context.with_supplied_terminal_chemistry(atoms, chemical_database, co)
    atoms = with_metal_coordination_typed(atoms)
    atoms, gaps = context._filter_supported_atoms_and_connectivity(
        atoms,
        co,
        filter_missing_mainchain=(filter_missing_mainchain and atom37_n_poses is None),
        cut_partners=cut_partners,
    )
    chains, labels, insertion_codes, residue_labels, names = (
        context._extract_residue_metadata(atoms, gaps)
    )
    atom_residues = struc.get_all_residue_positions(atoms)
    type_index = {name: i for i, name in enumerate(co.restype_io_equiv_classes)}
    types = np.asarray([type_index[name] for name in names], dtype=np.int32)
    valid, slots, residues = map_atoms_to_canonical(
        co, atom_residues, atoms.res_name, atoms.atom_name, atoms.element
    )
    if PROTONATION_VARIANT in atoms.get_annotation_categories():
        map_neutral_terminal_hydrogens(
            co,
            atom_residues,
            atoms.res_name,
            atoms.atom_name,
            atoms.get_annotation(PROTONATION_VARIANT),
            valid,
            slots,
        )
    template = template_array(atoms)
    bonds, disulfides, inferred = covalent_bonds_from_biotite(
        template, co, atom_residues, types, valid, slots
    )
    metals = metal_coordination_from_biotite(template, atom_residues, valid, slots)
    coords = None
    if atom37_n_poses is None:
        raw = atoms.coord
        if raw.ndim == 2:
            raw = raw[None]
        n_poses = len(raw)
        coords = np.full(
            (n_poses, len(names), co.max_n_canonical_atoms, 3), np.nan, np.float32
        )
        coords[:, residues, slots] = raw[:, valid]
    else:
        n_poses = atom37_n_poses
    if EXPLICIT_TERMINI in atoms.get_annotation_categories():
        stated = atoms.get_annotation(EXPLICIT_TERMINI)[struc.get_residue_starts(atoms)]
        closed = (chains[:-1] == chains[1:]) & (
            ((stated[:-1] & 2) != 0) | ((stated[1:] & 1) != 0)
        )
        gaps[:-1, 1] |= closed
        gaps[1:, 0] |= closed
    if coords is not None:
        bonds = break_polymer_gaps(
            gaps,
            bonds,
            inferred,
            coords,
            types,
            chains,
            co,
            missing_density_distance_threshold,
        )
    orient_polymer_gap_flags(gaps, chains, types, bonds, co)

    def repeat(values):
        return np.repeat(values[None], n_poses, axis=0)

    fields = dict(
        coords=coords,
        chain_id=repeat(chains.astype(np.int32)),
        res_types=repeat(types),
        chain_labels=repeat(labels.astype(object)),
        res_labels=repeat(residue_labels),
        residue_insertion_codes=repeat(insertion_codes.astype(object)),
        res_not_connected=repeat(gaps),
        covalent_bonds=bonds_for_poses(bonds, n_poses),
        disulfides=bonds_for_poses(disulfides, n_poses),
        metal_coordination=bonds_for_poses(metals, n_poses),
    )
    fields.update(
        _optional_metadata(
            atoms, n_poses, len(names), co.max_n_canonical_atoms, valid, residues, slots
        )
    )
    return CanonicalAtomArrayPlan(atoms, names, valid, residues, slots, fields)
