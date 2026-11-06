"""Shared Atom37 atom identity, slot routing and topology validation."""

import numpy
import biotite.structure
from tmol_core.io.canonical_ordering import CanonicalOrdering


class Atom37MappingError(ValueError):
    """An AtomArray cannot be routed unambiguously into an Atom37 tensor."""


def required_mainchain_entries(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    co: CanonicalOrdering,
) -> tuple[tuple[int, int, str], ...]:
    """Describe canonical mainchain atoms required for pose construction."""
    starts = biotite.structure.get_residue_starts(biotite_structure)
    residue_names = biotite.structure.get_residues(biotite_structure)[1]
    entries = []
    for residue, (start, residue_name) in enumerate(zip(starts, residue_names)):
        required = co.restypes_required_mainchain_atoms.get(residue_name) or ()
        mapping = co.restypes_atom_index_mapping[residue_name]
        label = (
            f"{residue_name} {biotite_structure.chain_id[start]}:"
            f"{biotite_structure.res_id[start]}"
            f"{biotite_structure.ins_code[start]}"
        )
        entries.extend(
            (residue, mapping[atom_name], f"{label}/{atom_name}")
            for atom_name in required
        )
    return tuple(entries)


def atom37_mapping(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    valid_atom_mask: numpy.ndarray,
    valid_res_inds: numpy.ndarray,
    valid_atom_inds: numpy.ndarray,
    max_n_tokens: int | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Validate and return source/target indices for Atom37 routing."""
    categories = set(biotite_structure.get_annotation_categories())
    missing = {"token_id", "atom37_slot"} - categories
    if missing:
        raise Atom37MappingError(
            "atom37 coordinate routing requires integer Biotite annotation(s): "
            + ", ".join(sorted(missing))
        )

    token_id = numpy.asarray(biotite_structure.token_id)
    slot = numpy.asarray(biotite_structure.atom37_slot)
    if not numpy.issubdtype(token_id.dtype, numpy.integer):
        raise Atom37MappingError(
            "Biotite token_id annotation must have an integer dtype"
        )
    if not numpy.issubdtype(slot.dtype, numpy.integer):
        raise Atom37MappingError(
            "Biotite atom37_slot annotation must have an integer dtype"
        )

    token_id = token_id.astype(numpy.int64, copy=False)[valid_atom_mask]
    slot = slot.astype(numpy.int64, copy=False)[valid_atom_mask]
    mapped = (token_id >= 0) & (slot >= 0)
    if numpy.any(slot[mapped] >= 37):
        maximum = int(slot[mapped].max())
        raise Atom37MappingError(
            f"atom37_slot values must be less than 37; got {maximum}"
        )
    if not numpy.any(mapped):
        raise Atom37MappingError("No supported Biotite atoms map to atom37_coords")
    if max_n_tokens is not None and numpy.any(token_id[mapped] >= max_n_tokens):
        maximum = int(token_id[mapped].max())
        raise Atom37MappingError(
            f"token_id {maximum} exceeds atom37_coords token count {max_n_tokens}"
        )

    source_pairs = numpy.column_stack((token_id[mapped], slot[mapped]))
    if numpy.unique(source_pairs, axis=0).shape[0] != source_pairs.shape[0]:
        raise Atom37MappingError(
            "Each mapped Biotite atom must use a unique (token_id, atom37_slot) pair"
        )
    return (
        token_id[mapped],
        slot[mapped],
        numpy.asarray(valid_res_inds)[mapped],
        numpy.asarray(valid_atom_inds)[mapped],
    )


def atom37_slot_map_for_ordering(canonical_ordering, atom_names_by_slot):
    """Build residue-specific canonical atom indices, preserving terminal policy."""
    n_slots = max((len(names) for names in atom_names_by_slot.values()), default=0)
    slot_map = numpy.full(
        (canonical_ordering.n_restype_io_equiv_classes, n_slots),
        -1,
        dtype=numpy.int64,
    )
    for restype_index, name3 in enumerate(canonical_ordering.restype_io_equiv_classes):
        names = atom_names_by_slot.get(name3)
        if names is None:
            continue
        # Per residue type, not pooled: H1, H2 and H3 are terminus-only on an
        # amino acid but ordinary base atoms of DG, DA, DT and water, and a
        # pooled set would leave those unrouted and silently rebuilt.
        terminus_atoms = canonical_ordering.termini_only_atoms_by_class.get(name3, ())
        atom_index = canonical_ordering.restypes_atom_index_mapping[name3]
        for slot, atom_name in enumerate(names):
            if not atom_name or atom_name in terminus_atoms:
                continue
            canonical_atom = atom_index.get(atom_name)
            if canonical_atom is not None:
                slot_map[restype_index, slot] = canonical_atom
    return slot_map
