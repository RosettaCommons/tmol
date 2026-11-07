"""Atom identity and connectivity across the same molecular domains."""

import numpy as np
import pytest

from .molecules import DOMAIN_CASES


@pytest.mark.parametrize("name", DOMAIN_CASES)
def test_explicit_molecular_atom_identity_and_connection_context(backend, name):
    case = DOMAIN_CASES[name]
    backend.molecule(case, terms=("ref",))
    pose = backend.pose
    actual = backend.coords()
    types = backend.numpy(pose.block_type_ind)
    offsets = backend.numpy(pose.block_coord_offset)
    connections = backend.numpy(pose.inter_residue_connections)
    palette = pose.packed_block_types.active_block_types
    for p, row in enumerate(case.blocks(backend.parameter_database)):
        expected = np.concatenate([b[2] for b in row])
        np.testing.assert_array_equal(actual[p, : len(expected)], expected)
        np.testing.assert_array_equal(actual[p, len(expected) :], 0)
        expected_connections = np.full_like(connections[p], -1)
        for b, (name, atom_names, coords) in enumerate(row):
            bt = palette[types[p, b]]
            assert bt.name == name
            assert tuple(atom.name for atom in bt.atoms) == atom_names
            assert offsets[p, b] == sum(len(block[1]) for block in row[:b])
        for left, port, right, other_port in case.connections()[p]:
            left_type, right_type = palette[types[p, left]], palette[types[p, right]]
            li = [c.name for c in left_type.connections].index(port)
            ri = [c.name for c in right_type.connections].index(other_port)
            expected_connections[left, li] = [right, ri]
            expected_connections[right, ri] = [left, li]
        np.testing.assert_array_equal(
            connections[p, ..., 0], expected_connections[..., 0]
        )
        # Native topology uses partner=-1 for a disconnected port; the second
        # slot is unspecified in that case (legacy Torch may retain zero).
        connected = expected_connections[..., 0] >= 0
        np.testing.assert_array_equal(
            connections[p][connected], expected_connections[connected]
        )
