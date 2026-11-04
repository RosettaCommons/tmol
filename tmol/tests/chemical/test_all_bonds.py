import numpy
import pytest

from tmol.chemical._all_bonds import _bonds_and_bond_ranges


@pytest.mark.parametrize("n_atoms", [0, 1, 4])
def test_no_bonds(n_atoms):
    bonds, ranges = _bonds_and_bond_ranges(
        n_atoms, numpy.empty((0, 2), dtype=numpy.int32), numpy.array([], numpy.int32)
    )
    assert bonds.shape == (0, 3)
    numpy.testing.assert_array_equal(ranges, numpy.zeros((n_atoms, 2), numpy.int32))


@pytest.mark.parametrize("connected_atoms", [(0, 1), (1, 2), (2, 3)])
@pytest.mark.parametrize("connections", [[], [1], [3, 3]])
def test_bond_ranges_with_isolated_atoms(connected_atoms, connections):
    a, b = connected_atoms
    intra = numpy.array([[a, b], [b, a]], dtype=numpy.int32)
    connections = numpy.array(connections, dtype=numpy.int32)
    bonds, ranges = _bonds_and_bond_ranges(5, intra, connections)
    expected = []
    for atom in range(5):
        start = len(expected)
        expected.extend([int(src), int(dst), -1] for src, dst in intra if src == atom)
        expected.extend(
            [atom, -1, index]
            for index, connection_atom in enumerate(connections)
            if connection_atom == atom
        )
        numpy.testing.assert_array_equal(ranges[atom], [start, len(expected)])
    numpy.testing.assert_array_equal(bonds, expected)
