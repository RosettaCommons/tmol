from types import SimpleNamespace as S

import numpy as np
import pytest

from tmol_core.io.polymer_gaps import break_polymer_gaps, orient_polymer_gap_flags


def test_ensemble_gaps_keep_unresolved_contacts_and_respect_declared_orientation():
    # Deliberately different class-specific ports; no peptide atom-name rule.
    co = S(
        polymer_conn_inds=S(
            up_atom_for_co_restype=[2, 1], down_atom_for_co_restype=[0, 3]
        )
    )
    types = np.array([0, 1, -1])
    chain = np.zeros(3, dtype=np.int64)
    xyz = np.zeros((2, 3, 4, 3))
    xyz[:, 1] = 20
    bonds = np.array([[0, 2, 1, 3]], dtype=np.int64)
    inferred = np.array([True])
    flags = np.zeros((3, 2), bool)
    xyz[1, 1, 3] = np.nan
    assert (
        len(break_polymer_gaps(flags, bonds, inferred, xyz, types, chain, co, 2.4)) == 1
    )
    assert not flags.any()
    xyz[1, 1, 3] = 20
    xyz[:, 1, 1] = 1  # Reverse contact cannot override the graph-specified ports.
    remaining = break_polymer_gaps(flags, bonds, inferred, xyz, types, chain, co, 2.4)
    assert remaining.shape == (0, 4)
    np.testing.assert_array_equal(flags, [[False, True], [True, False], [False, False]])
    with pytest.raises(ValueError, match="Declared polymer bond"):
        break_polymer_gaps(flags, bonds, ~inferred, xyz, types, chain, co, 2.4)
    assert (
        break_polymer_gaps(flags, bonds, ~inferred, xyz, types, chain, co, 0) is bonds
    )


@pytest.mark.parametrize("container", [list, np.asarray])
def test_reverse_chain_orientation_ignores_padding_and_other_chains(container):
    co = S(
        polymer_conn_inds=S(up_atom_for_co_restype=[2], down_atom_for_co_restype=[0])
    )
    types = container([0, 0, -1, 0, 0])
    chains = np.array([0, 0, 0, 1, 1])
    bonds = np.array([[0, 0, 1, 2], [3, 2, 4, 0]])
    flags = np.array(
        [[False, True], [True, False], [False, False], [False, True], [False, False]]
    )
    orient_polymer_gap_flags(flags, chains, types, bonds, co)
    np.testing.assert_array_equal(
        flags,
        [[True, False], [False, True], [False, False], [False, True], [False, False]],
    )


def test_padding_never_borrows_the_last_database_class():
    co = S(polymer_conn_inds=S(up_atom_for_co_restype=[], down_atom_for_co_restype=[]))
    flags = np.zeros((2, 2), bool)
    bonds = np.empty((0, 4), np.int64)
    out = break_polymer_gaps(
        flags,
        bonds,
        np.empty(0, bool),
        np.zeros((1, 2, 0, 3)),
        np.array([-1, -1]),
        np.zeros(2),
        co,
        2.4,
    )
    assert out.shape == (0, 4)
    assert not flags.any()
