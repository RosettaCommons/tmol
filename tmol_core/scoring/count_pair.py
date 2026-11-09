"""Atom-typed count-pair preparation for LJ/LK and electrostatics.

Bond distances are chemistry, while these encoded distances select the
nonbonded weighting rule. Hydrogen bonds continue to use the original distances.
"""

import numpy as np


def ligand_atom_mask(block_type, rosetta_typed):
    """Which physical atom types use the generic ligand count-pair convention."""
    return np.array(
        [atom.atom_type not in rosetta_typed for atom in block_type.atoms], dtype=bool
    )


def encode_intra_distances(distances, ligand_atoms, *, copy=True):
    """Encode generic/generic 1–4 and 1–5 pairs as full weight.

    Atom axes follow the physical atoms even when distances use charge
    representatives. Canonical/generic pairs retain the canonical rule.
    The default owns a copy; callers may reuse a fresh writable array.
    """
    encoded = np.array(distances, copy=copy)
    both = ligand_atoms[:, None] & ligand_atoms[None, :]
    encoded[both & ((encoded == 3) | (encoded == 4))] = 5
    return encoded


def electrostatic_distances(block_type, representatives, rosetta_typed):
    """Return inter/intra representative-distance tables without changing topology."""
    inter = block_type.path_distance[:, representatives]
    intra = encode_intra_distances(
        inter[representatives, :],
        ligand_atom_mask(block_type, rosetta_typed),
        copy=False,
    )
    return inter, intra


def packed_ljlk_distances(block_types, max_n_atoms, rosetta_typed):
    """Host table and all-generic flags in active block order, with distance-6 padding."""
    distances = np.full((len(block_types), max_n_atoms, max_n_atoms), 6, dtype=np.int32)
    all_ligand = np.zeros(len(block_types), dtype=np.int32)
    for index, block in enumerate(block_types):
        ligand = ligand_atom_mask(block, rosetta_typed)
        distances[index, : block.n_atoms, : block.n_atoms] = encode_intra_distances(
            block.path_distance, ligand
        )
        all_ligand[index] = ligand.all()
    return distances, all_ligand
