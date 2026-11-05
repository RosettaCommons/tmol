"""Native bond metadata for arbitrary refined residue palettes."""

import numpy
from .constants import MAX_SIG_BOND_SEPARATION


def pack_bond_separations(block_types, max_n_atoms, padding=MAX_SIG_BOND_SEPARATION):
    blocks = tuple(block_types)
    result = numpy.full(
        (len(blocks), max_n_atoms, max_n_atoms), padding, dtype=numpy.int32
    )
    for i, block in enumerate(blocks):
        result[i, : block.n_atoms, : block.n_atoms] = block.path_distance
    return result


def pack_bond_topology(block_types, max_n_atoms=None):
    """Pack real intra-residue bonds and unresolved inter-residue connections.

    Preserve the canonical connection records from refined chemistry, including
    their ordering, and retain -1 padding for absent bonds and atom ranges.
    """
    blocks = tuple(block_types)
    if max_n_atoms is None:
        max_n_atoms = max((b.n_atoms for b in blocks), default=0)
    max_n_all_bonds = max((b.all_bonds.shape[0] for b in blocks), default=0)
    n_all_bonds = numpy.full((len(blocks),), -1, dtype=numpy.int32)
    all_bonds = numpy.full((len(blocks), max_n_all_bonds, 3), -1, dtype=numpy.int32)
    atom_all_bond_ranges = numpy.full(
        (len(blocks), max_n_atoms, 2), -1, dtype=numpy.int32
    )
    for i, block in enumerate(blocks):
        count = block.all_bonds.shape[0]
        n_all_bonds[i] = count
        all_bonds[i, :count] = block.all_bonds
        atom_all_bond_ranges[i, : block.n_atoms] = block.atom_all_bond_ranges
    return dict(
        bond_separation=pack_bond_separations(blocks, max_n_atoms),
        n_all_bonds=n_all_bonds,
        all_bonds=all_bonds,
        atom_all_bond_ranges=atom_all_bond_ranges,
    )
