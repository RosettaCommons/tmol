import numpy as np
import pytest

from tmol_core.pose.constraints import ConstraintTopology


def test_constraint_batch_split_preserves_atom_identity_and_row_order():
    # Nonconsecutive block IDs and large atom indices are just topology labels.
    atoms = np.array([[[8, 301], [12, 511]], [[12, 1], [12, 17]]], np.int32)
    topology, repeats = ConstraintTopology.empty(3).add(atoms, 4, replace_empty=True)
    assert repeats == 3
    pieces = [topology.split(i) for i in range(3)]
    for i, (piece, rows) in enumerate(pieces):
        np.testing.assert_array_equal(rows, [2 * i, 2 * i + 1])
        np.testing.assert_array_equal(piece.constraint_atoms[:, :2, 1:], atoms)
        np.testing.assert_array_equal(piece.constraint_num_unique_blocks, [2, 1])
    joined = ConstraintTopology.concatenate(
        [piece for piece, _ in pieces], [np.arange(5)] * 3
    )
    for name, values in topology.arrays().items():
        np.testing.assert_array_equal(getattr(joined, name), values)
    # The missing middle atom slots remain sentinels; the final slot retains
    # the last atom for score attribution, including after pose-index shifts.
    assert np.all(joined.constraint_atoms[:, 2] == -1)
    np.testing.assert_array_equal(
        joined.constraint_atoms[:, 3], joined.constraint_atoms[:, 1]
    )


@pytest.mark.parametrize("n_poses", [0, 2])
def test_empty_constraint_batch_has_defined_padded_shapes(n_poses):
    topology, repeats = ConstraintTopology.empty(n_poses).add(
        np.empty((0, 2, 2), np.int32), 0
    )
    assert repeats == n_poses
    assert topology.constraint_atoms.shape == (0, 4, 3)
    piece, rows = topology.split(0)
    assert rows.shape == (0,)
    assert piece.constraint_unique_blocks.shape == (0, 3)


@pytest.mark.parametrize(
    "indices,n_poses",
    [
        ([[[0, 0, -1]]], 1),
        ([[[0, 0, 2**31]]], 1),
        ([[[1, 0, 1]]], 1),
        ([[[0, 0, 1]]], 0),
    ],
)
def test_constraint_input_indices_reject_wraparound_and_invalid_poses(indices, n_poses):
    with pytest.raises(ValueError, match="indices"):
        ConstraintTopology.empty(n_poses).add(np.array(indices, np.int64), 0)


@pytest.mark.parametrize("n_poses", [-1, 1.5, 2**31])
def test_constraint_pose_count_is_a_nonnegative_int32(n_poses):
    with pytest.raises(ValueError, match="n_poses"):
        ConstraintTopology.empty(n_poses)


def test_shared_flatbottom_coordinate_translation_and_zero_region():
    from tmol_core.scoring.constraints import ConstraintOps, coordinate_flatbottom

    ops = ConstraintOps(
        lambda v: np.linalg.norm(v, axis=-1), np.where, np.trunc, np.sign, None
    )
    atoms = np.array([[[0.0, 0.0, 0.0]], [[3.0, 4.0, 0.0]]])
    params = np.array([[0.0, 0.0, 0.0, 2.0, 1.0]] * 2)
    np.testing.assert_array_equal(
        coordinate_flatbottom(atoms, params, ops), [0.0, 16.0]
    )
    shift = np.array([100.0, -37.0, 82.0])
    translated = params.copy()
    translated[:, :3] += shift
    np.testing.assert_array_equal(
        coordinate_flatbottom(atoms + shift, translated, ops), [0.0, 16.0]
    )


@pytest.mark.parametrize(
    "blocks,expected", [([8, 12, 8, 12], 2), ([8, 12, 3, 8], 3), ([8, 8, 8, 8], 1)]
)
def test_constraint_body_count_ignores_atom_order(blocks, expected):
    atoms = np.zeros((1, 4, 3), np.int32)
    atoms[0, :, 1] = blocks
    topology, _ = ConstraintTopology.empty(1).add(atoms, 0)
    np.testing.assert_array_equal(topology.constraint_num_unique_blocks, [expected])
    np.testing.assert_array_equal(topology.constraint_atoms, atoms)
