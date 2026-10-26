"""Per-residue-type all-bonds table: intra-block bonds merged with the
inter-block edges implied by the residue's ``connections``. Ported from
``tmol.chemical.all_bonds`` (pure numpy, no numba).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def bonds_and_bond_ranges(
    n_atoms: int,
    intra_res_bonds: np.ndarray,  # int32[:, 2]
    ordered_connection_atoms: np.ndarray,  # int32[:]
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate the set of intra- and inter-block bonds.

    The intra-block bonds should list each bond twice, once for each of the
    two atoms it connects. The ``ordered_connection_atoms`` array lists the
    atom on this block that connects to another block such that the ith
    position in this array represents the ith inter-block connection; the
    case when a single atom serves as a connection point to multiple other
    blocks (as might happen with metal ions) is handled correctly.

    Returns
    -------
    bonds: ``int32[n_all_bonds, 3]``
        Each row is ``(atom_ind, other_atom_ind_or_-1, conn_id_or_-1)``.
        If the partner is on this residue, ``other_atom_ind`` is the local
        index and ``conn_id`` is ``-1``. Otherwise ``other_atom_ind`` is
        ``-1`` and ``conn_id`` is the index into this residue's
        ``connections`` tuple.
    bond_ranges: ``int32[n_atoms, 2]``
        Start/end row indices in ``bonds`` for each atom. Bonds of a given
        atom are contiguous in ``bonds``.
    """
    int_dtype = intra_res_bonds.dtype
    n_intra_bonds = int(intra_res_bonds.shape[0])
    n_inter_bonds = int(ordered_connection_atoms.shape[0])
    n_all_bonds = n_intra_bonds + n_inter_bonds

    count_intra_bonds = np.zeros((n_atoms,), dtype=int_dtype)
    count_inter_bonds = np.zeros((n_atoms,), dtype=int_dtype)

    for i in range(n_intra_bonds):
        count_intra_bonds[intra_res_bonds[i, 0]] += 1
    for i in range(n_inter_bonds):
        count_inter_bonds[ordered_connection_atoms[i]] += 1

    bond_ranges = np.full((n_atoms, 2), -1, dtype=int_dtype)
    for i in range(n_atoms):
        i_n_bonds = count_intra_bonds[i] + count_inter_bonds[i]
        if i == 0:
            if i_n_bonds > 0:
                bond_ranges[i, 0] = 0
                bond_ranges[i, 1] = i_n_bonds
            else:
                # No bonds for atom 0 -- keep a valid (start == end) range
                bond_ranges[i, 0] = 0
                bond_ranges[i, 1] = 0
        else:
            bond_ranges[i, 0] = bond_ranges[i - 1, 1]
            bond_ranges[i, 1] = bond_ranges[i - 1, 1] + i_n_bonds

    bonds = np.zeros((n_all_bonds, 3), dtype=int_dtype)
    count_bond = np.zeros((n_atoms,), dtype=int_dtype)
    for i in range(n_intra_bonds):
        i_atm = int(intra_res_bonds[i, 0])
        i_count = int(count_bond[i_atm])
        count_bond[i_atm] += 1
        bond = int(bond_ranges[i_atm, 0]) + i_count
        bonds[bond, 0] = i_atm
        bonds[bond, 1] = intra_res_bonds[i, 1]
        bonds[bond, 2] = -1
    for i in range(n_inter_bonds):
        i_atm = int(ordered_connection_atoms[i])
        i_count = int(count_bond[i_atm])
        count_bond[i_atm] += 1
        bond = int(bond_ranges[i_atm, 0]) + i_count
        bonds[bond, 0] = i_atm
        bonds[bond, 1] = -1  # unresolved; on another residue
        bonds[bond, 2] = i  # i = index of inter-residue connection

    return bonds, bond_ranges


# --------------------------------------------------------------------- #
# PoseStack-level all-bonds                                              #
# --------------------------------------------------------------------- #


def _as_np_int(x) -> np.ndarray:
    """Coerce a JAX / numpy / torch-like array to numpy ``int64``."""
    arr = np.asarray(x)
    if arr.dtype.kind not in ("i", "u"):
        arr = arr.astype(np.int64)
    return arr


__all__ = [
    "bonds_and_bond_ranges",
]
