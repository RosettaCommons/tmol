"""Molecular input preserves chemistry, atom identity, graph edges and precision."""

import dataclasses

import jax
import numpy as np
import pytest

from tmol_jax.database import ParameterDatabase
from tmol_jax.io import ResidueBlock, pose_stack_from_residue_blocks


@pytest.fixture(scope="module")
def database():
    return ParameterDatabase.get_default()


def block(database, name, dtype=np.float32):
    residue = next(r for r in database.chemical.residues if r.name == name)
    # Labeled, distinct coordinates make atom-order loss observable independently
    # of the ideal-coordinate implementation. These are not scoring fixtures.
    xyz = np.arange(3 * len(residue.atoms), dtype=dtype).reshape(-1, 3) / 7
    return ResidueBlock(name, [a.name for a in residue.atoms][::-1], xyz[::-1])


def test_mixed_batch_preserves_exact_types_atom_order_and_context(database):
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


def test_explicit_cycle_has_symmetric_connections_and_graph_distance(database):
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


@pytest.mark.parametrize(
    "edges",
    [
        [(0, "missing", 1, "down")],
        [(0, "up", 3, "down")],
        [(0, "up", 0, "down")],
        [(0, "up", 1, "down"), (0, "up", 2, "down")],
    ],
)
def test_invalid_connections_raise(database, edges):
    with pytest.raises(ValueError):
        pose_stack_from_residue_blocks(
            [[block(database, "GLY")] * 3],
            parameter_database=database,
            connections=[edges],
        )


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "unknown"])
def test_atom_and_residue_identity_cannot_be_silently_coerced(database, change):
    b = block(database, "DALA")
    if change == "missing":
        b = dataclasses.replace(b, atom_names=b.atom_names[:-1], coords=b.coords[:-1])
    elif change == "extra":
        b = dataclasses.replace(
            b,
            atom_names=[*b.atom_names, "UNKNOWN"],
            coords=np.concatenate([b.coords, b.coords[:1]]),
        )
    elif change == "duplicate":
        b = dataclasses.replace(b, atom_names=[b.atom_names[1], *b.atom_names[1:]])
    else:
        b = dataclasses.replace(b, residue_type="UNKNOWN")
    with pytest.raises(ValueError):
        pose_stack_from_residue_blocks([[b]], parameter_database=database)


def test_float64_coordinates_are_preserved_or_rejected_explicitly(database):
    inputs = [[block(database, "DALA", np.float64)]]
    with jax.enable_x64(True):
        pose = pose_stack_from_residue_blocks(inputs, parameter_database=database)
        assert pose.coords.dtype == np.float64
        np.testing.assert_array_equal(pose.coords[0], inputs[0][0].coords[::-1])
    with jax.enable_x64(False), pytest.raises(ValueError, match="float64"):
        pose_stack_from_residue_blocks(inputs, parameter_database=database)
