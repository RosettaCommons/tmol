"""Execution feedback must reach subsequent connected build waves."""

from types import SimpleNamespace as NS

import numpy as np

from tmol_core.io.context import build_missing_by_context


def test_connected_context_uses_executed_coordinates_without_mutating_inputs():
    ideal = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 1]], float)
    bonds = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]])

    def block(connection_atom):
        return NS(
            n_atoms=4,
            atoms=[NS(name=f"A{i}", atom_type="C") for i in range(4)],
            ideal_coords=ideal,
            icoors_index={**{f"A{i}": i for i in range(4)}, "link": 4},
            atom_to_idx={f"A{i}": i for i in range(4)},
            bond_indices=bonds,
            bond_to_type={tuple(pair): 1 for pair in bonds.tolist()},
            bond_to_ringness={},
            connections=[NS(atom=f"A{connection_atom}", name="link")],
            connection_bond_types=[1],
            connection_bond_in_ring=[False],
            chi_samples=[],
            torsions=[],
        )

    atom_types = {
        "C": NS(
            element="C",
            is_donor=False,
            is_acceptor=False,
            is_metal=False,
            is_metal_donor=False,
        )
    }
    coordinates = np.concatenate((ideal[:4], ideal[:4] + 4))[None].astype(np.float32)
    before = coordinates.copy()
    targets = np.array([[False, False, False, True] * 2])
    feedback = np.array([1.123456789, 1.234567891, 1.345678912], dtype=np.float32)
    visited = []

    def place(build):
        visited.append(build.b)
        assert build.built == [3]
        if build.b == 1:
            np.testing.assert_array_equal(build.node_xyz[4], feedback.astype(float))
        return {3: feedback + 4 * build.b}

    built = build_missing_by_context(
        [block(3), block(0)],
        atom_types,
        np.array([[3], [0]]),
        coordinates,
        targets,
        targets,
        np.array([[0, 4]]),
        np.array([[0, 1]]),
        np.array([[[[1, 0]], [[0, 0]]]]),
        place,
    )
    assert visited == [0, 1]
    np.testing.assert_array_equal(built, targets)
    np.testing.assert_array_equal(coordinates, before)
