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

    if backend.framework != "jax" or case is not DOMAIN_CASES["mixed_batch"]:
        return
    import dataclasses
    from tmol_jax.database import ParameterDatabase
    from tmol_jax.io import ResidueBlock, pose_stack_from_residue_blocks

    jax = backend.module
    database = ParameterDatabase.get_default()

    def block(database, name, dtype=np.float32):
        residue = next(r for r in database.chemical.residues if r.name == name)
        # Labeled, distinct coordinates make atom-order loss observable independently
        # of the ideal-coordinate implementation. These are not scoring fixtures.
        xyz = np.arange(3 * len(residue.atoms), dtype=dtype).reshape(-1, 3) / 7
        return ResidueBlock(name, [a.name for a in residue.atoms][::-1], xyz[::-1])

    with jax.default_device(backend.device):
        inputs = [
            [block(database, name) for name in ("DALA", "HOH")],
            [block(database, "GLY")],
        ]
        pose, context = pose_stack_from_residue_blocks(
            inputs, parameter_database=database, return_context=True
        )
        assert context.parameter_database is database
        assert pose.packed_block_types.chem_db is database.chemical
        assert [
            [pose.block_type(p, b).name for b in range(len(row))]
            for p, row in enumerate(inputs)
        ] == [["DALA", "HOH"], ["GLY"]]
        for p, row in enumerate(inputs):
            expected = np.concatenate([r.coords[::-1] for r in row])
            np.testing.assert_array_equal(pose.coords[p, : len(expected)], expected)
            assert np.asarray(pose.atom_is_real[p]).sum() == len(expected)
        assert np.all(np.asarray(pose.inter_residue_connections) == -1)
        reordered = pose_stack_from_residue_blocks(inputs[::-1], context=context)
        assert reordered.packed_block_types is context.packed_block_types
        assert reordered.block_type(0, 0).name == "GLY"
        inputs = [[block(database, "GLY") for _ in range(3)]]
        edges = [[(0, "up", 1, "down"), (1, "up", 2, "down"), (2, "up", 0, "down")]]
        pose = pose_stack_from_residue_blocks(
            inputs, parameter_database=database, connections=edges
        )
        rt = pose.block_type(0, 0)
        up, down = rt.connection_to_cidx["up"], rt.connection_to_cidx["down"]
        conn, sep = map(
            np.asarray, (pose.inter_residue_connections, pose.inter_block_bondsep)
        )
        for left, _, right, _ in edges[0]:
            np.testing.assert_array_equal(conn[0, left, up], [right, down])
            np.testing.assert_array_equal(conn[0, right, down], [left, up])
            assert sep[0, left, right, up, down] == 1
        # GLY down/up ports lie on N and C: N--CA--C is exactly two bonds.
        assert sep[0, 0, 0, down, up] == 2
        # N0--C2--CA2--N2 takes three bonds across the closing edge.
        assert sep[0, 0, 2, down, down] == 3
        for edges in (
            [(0, "missing", 1, "down")],
            [(0, "up", 3, "down")],
            [(0, "up", 0, "down")],
            [(0, "up", 1, "down"), (0, "up", 2, "down")],
        ):
            with pytest.raises(ValueError):
                pose_stack_from_residue_blocks(
                    [[block(database, "GLY")] * 3],
                    parameter_database=database,
                    connections=[edges],
                )
        for change in ("missing", "extra", "duplicate", "unknown"):
            b = block(database, "DALA")
            if change == "missing":
                b = dataclasses.replace(
                    b, atom_names=b.atom_names[:-1], coords=b.coords[:-1]
                )
            elif change == "extra":
                b = dataclasses.replace(
                    b,
                    atom_names=[*b.atom_names, "UNKNOWN"],
                    coords=np.concatenate([b.coords, b.coords[:1]]),
                )
            elif change == "duplicate":
                b = dataclasses.replace(
                    b, atom_names=[b.atom_names[1], *b.atom_names[1:]]
                )
            else:
                b = dataclasses.replace(b, residue_type="UNKNOWN")
            with pytest.raises(ValueError):
                pose_stack_from_residue_blocks([[b]], parameter_database=database)
        inputs = [[block(database, "DALA", np.float64)]]
        with jax.enable_x64(True):
            pose = pose_stack_from_residue_blocks(inputs, parameter_database=database)
            assert pose.coords.dtype == np.float64
            np.testing.assert_array_equal(pose.coords[0], inputs[0][0].coords[::-1])
        with jax.enable_x64(False), pytest.raises(ValueError, match="float64"):
            pose_stack_from_residue_blocks(inputs, parameter_database=database)
