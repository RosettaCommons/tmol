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


def index_bonds(src_bonds, minlength=None):
    # Convert undirected (stack, i, j) bond index tuples into sorted, indexed list.
    if len(src_bonds) == 0:
        return dict(
            bonds=numpy.empty((1, 0, 2), dtype=numpy.int64),
            bond_spans=numpy.zeros((1, minlength or 0, 2), dtype=numpy.int64),
        )

    uniq_bonds, src_index = numpy.unique(src_bonds, axis=0, return_index=True)

    nstacks = numpy.max(uniq_bonds[:, 0]) + 1

    if not minlength:
        minlength = numpy.max(uniq_bonds[:, 1:]) + 1

    bond_counts_per_stack = numpy.bincount(uniq_bonds[:, 0], minlength=nstacks).reshape(
        (nstacks, 1)
    )
    max_nbonds = numpy.max(bond_counts_per_stack)

    bonds = numpy.full((nstacks, max_nbonds, 2), -1, dtype=int)
    counts = numpy.arange(max_nbonds, dtype=int).reshape((1, max_nbonds))
    lowinds = counts < bond_counts_per_stack
    nz = numpy.nonzero(lowinds)
    bonds[nz[0], nz[1], :] = uniq_bonds[:, 1:]

    bond_spans = numpy.full((nstacks, minlength, 2), -1, dtype=int)

    # bond spans: the starting and stoping index for every bond for an atom;
    # created using the cumsum function after counting how many bonds each
    # atom has, using the bincount function.
    #
    # To use the bincount function, we need to ravel the 1st-atom-in-bond
    # array, and we need to assign atoms in separate stacks to diferent
    # integers (which is what max_atom_offsets is for). We also need to
    # set all of the negative bond inds to an out-of-range value (like
    # (maxats+1)*nstacks -1) and then, after performing the bincount,
    # slice off the part of the array that carries its counts.

    max_atom_offsets = numpy.arange(nstacks, dtype=int).reshape((nstacks, 1)) * (
        minlength + 1
    )
    first_at_shifted = numpy.copy(bonds[:, :, 0])
    first_at_shifted[first_at_shifted < 0] = (minlength + 1) * nstacks - 1
    first_at_shifted = (first_at_shifted + max_atom_offsets).ravel()
    bincounts = numpy.bincount(
        first_at_shifted, minlength=((minlength + 1) * nstacks)
    ).reshape(nstacks, -1)[:, :minlength]
    bonds_cumsum = numpy.cumsum(bincounts, axis=1)
    bond_spans[:, 0, 0] = 0
    bond_spans[:, 1:minlength, 0] = bonds_cumsum[:, :-1]
    bond_spans[:, 0:minlength, 1] = bonds_cumsum

    return dict(bonds=bonds, bond_spans=bond_spans)


def directed_bonds(src_bonds):
    """Add reversed edges to (batch, atom, atom) rows, preserving input order."""
    return numpy.concatenate(
        (
            src_bonds,
            numpy.concatenate(
                (src_bonds[:, 0:1], src_bonds[:, -1:], src_bonds[:, -2:-1]), axis=-1
            ),
        ),
        axis=0,
    )
