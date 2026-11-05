"""Runtime transforms for the shared native coordinate-to-DOF operation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.kinematics._native import inverse_coordinates
from tmol_jax.kinematics import KinematicForest


def test_host_metadata_transport_preserves_inputs_and_owns_outputs():
    from tmol_jax.kinematics._host import NativePosePlanner

    # Strided int64 inputs exercise conversion without rewriting user metadata.
    offsets = np.array([[0, -8, 2, -8]], dtype=np.int64)[:, ::2]
    types = np.array([[0, 0]], dtype=np.int64)
    counts = np.array([2], dtype=np.int64)
    real = np.ones((1, 2), dtype=bool)
    inputs = (offsets, types, counts, real)
    snapshots = tuple(value.copy() for value in inputs)
    planner = NativePosePlanner()
    first = planner.get_kfo_indices_for_atoms(*inputs)
    saved = tuple(value.copy() for value in first)
    # Another allocation cannot invalidate the preceding released native result.
    second = planner.get_kfo_indices_for_atoms(*inputs)
    for value, snapshot in zip(inputs, snapshots):
        np.testing.assert_array_equal(value, snapshot)
    for value, expected, other in zip(first, saved, second):
        assert value.dtype == np.int32 and not np.shares_memory(value, other)
        np.testing.assert_array_equal(value, expected)
    for malformed, error in (
        (offsets.astype(float), TypeError),
        (offsets + (1 << 32), OverflowError),
        (offsets[None], ValueError),
    ):
        with pytest.raises(error):
            planner.get_kfo_indices_for_atoms(malformed, *inputs[1:])
    with pytest.raises(ValueError, match="missing native host array"):
        planner.get_kfo_indices_for_atoms(*inputs[:-1])
    with pytest.raises(ValueError, match="extra native host arguments"):
        planner.get_kfo_indices_for_atoms(*inputs, real)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inverse_jump_translation_batching_and_initialization_boundary(dtype):
    # Two independent rigid-body jumps from the virtual origin. Near-collinear
    # frame points exercise the common frame repair, without repeating its math.
    with jax.enable_x64():
        xyz = jnp.asarray([[0, 0, 0], [1, 0, 0], [2, 1e-4, 0]], dtype=dtype)
        arrays = tuple(
            np.asarray(row, dtype=np.int32)
            for row in ([0, 0, 0], [0, 1, 2], [0, 0, 1], [0, 2, 0], [0, 1, 1])
        )
        forest = KinematicForest(*arrays, device=xyz.device)
        inverse = jax.jit(forest.inverse)
        dofs = inverse(xyz)
        assert forest.n_nodes == 3 and dofs.device == forest.device
        assert dofs.dtype == xyz.dtype
        assert np.isfinite(dofs).all()
        np.testing.assert_array_equal(dofs[0], 0)
        np.testing.assert_array_equal(dofs[1:, :3], xyz[1:])
        np.testing.assert_array_equal(dofs[:, 3:6], 0)
        batched = jax.jit(jax.vmap(inverse))(jnp.stack((xyz, 2 * xyz)))
        np.testing.assert_array_equal(batched[0], dofs)
        np.testing.assert_array_equal(batched[1, 1:, :3], 2 * xyz[1:])
        np.testing.assert_allclose(batched[1, :, 6:], dofs[:, 6:], atol=1e-6)
        with pytest.raises(NotImplementedError, match="initializes independent"):
            jax.grad(lambda x: inverse(x).sum())(xyz)


def test_inverse_empty_forest_and_invalid_runtime_layout():
    forest = (jnp.zeros(0, jnp.int32),) * 5
    xyz = jnp.zeros((0, 3), jnp.float32)
    result = jax.jit(lambda x: inverse_coordinates(x, forest))(xyz)
    assert result.shape == (0, 9)
    with pytest.raises(TypeError, match="float32 or float64"):
        inverse_coordinates(xyz.astype(jnp.int32), forest)
    with pytest.raises(ValueError, match="shape"):
        inverse_coordinates(xyz.reshape(0), forest)
    with pytest.raises(ValueError, match="five int32"):
        inverse_coordinates(xyz, forest[:4])


def test_prepared_forest_snapshots_indices_and_uses_common_validation():
    arrays = [
        np.array(row, dtype=np.int32)
        for row in ([0, 0, 0], [0, 1, 2], [0, 0, 1], [0, 2, 0], [0, 1, 1])
    ]
    forest = KinematicForest(*arrays)
    xyz = jnp.array([[0, 0, 0], [1, 0, 0], [2, 0.1, 0]], dtype=jnp.float32)
    expected = forest.inverse(xyz)
    arrays[0][1:] = [2, 1]
    np.testing.assert_array_equal(forest.inverse(xyz), expected)
    with pytest.raises(ValueError, match="cycle"):
        KinematicForest(*arrays)


def test_bound_forward_transform_contract_and_snapshot():
    from tests.scientific.kinematics import branched_forest

    parent, kinds, reference = branched_forest(np.float32)
    zeros = np.zeros(len(parent), np.int32)
    forest = KinematicForest(parent, zeros, zeros, zeros, kinds)
    mapping = forest.bind(reference)
    forward = jax.jit(mapping)
    initial = mapping.initial_dofs
    expected = forward(initial)
    assert expected.device == forest.device
    # Atom storage need not be a topological ordering of the parent tree.
    permutation = np.array([0, 4, 1, 5, 3, 2])
    inverse_permutation = np.argsort(permutation)
    reordered = KinematicForest(
        inverse_permutation[parent[permutation]],
        zeros,
        zeros,
        zeros,
        kinds[permutation],
    ).bind(reference[permutation])
    np.testing.assert_allclose(
        jax.jit(reordered)(reordered.initial_dofs),
        expected[permutation],
        rtol=1e-6,
        atol=1e-6,
    )
    reference[:, 6:] += 10
    np.testing.assert_array_equal(forward(initial), expected)
    shifted = initial.at[1, 0].add(2)
    batch = jax.jit(jax.vmap(mapping))(jnp.stack((initial, shifted)))
    np.testing.assert_array_equal(batch[0], expected)
    np.testing.assert_allclose(
        batch[1, 1:], expected[1:] + jnp.array([2, 0, 0]), atol=1e-6
    )
    gradients = jax.jit(jax.vmap(jax.grad(lambda x: mapping(x).sum())))(
        jnp.stack((initial, shifted))
    )
    np.testing.assert_allclose(gradients[0], gradients[1], atol=3e-6)
    with pytest.raises(TypeError, match="forward-mode autodiff"):
        jax.jvp(mapping, (initial,), (jnp.ones_like(initial),))
    with pytest.raises(ValueError, match="cannot be differentiated"):
        jax.jacrev(jax.grad(lambda x: mapping(x).sum()))(initial)
    with pytest.raises(ValueError, match="nodes, 6"):
        mapping(jnp.zeros((len(parent), 9)))
    with pytest.raises(TypeError, match="dtype"):
        mapping(initial.astype(jnp.int32))
    with pytest.raises(ValueError, match="nodes, 9"):
        forest.bind(reference[:, :6])


def test_bound_empty_forward_and_reverse():
    forest = KinematicForest(*(np.zeros(0, np.int32),) * 5)
    mapping = forest.bind(np.zeros((0, 9), np.float32))
    result = jax.jit(mapping)(mapping.initial_dofs)
    assert result.shape == (0, 3)
    gradient = jax.jit(jax.grad(lambda x: mapping(x).sum()))(mapping.initial_dofs)
    assert gradient.shape == (0, 6)


def test_fold_forest_chain_gaps_preserve_explicit_labels():
    from tmol_jax import FoldForest
    from tests.scientific.execution import Execution
    from tests.scientific.molecules import MolecularCase
    from tmol_core.kinematics.fold_forest import EdgeType

    pose = (
        Execution("jax-cpu")
        .molecule(MolecularCase((), (("ALA", "GLY"),), ((1, 1),)), terms=("ref",))
        .pose
    )
    separate = FoldForest.reasonable_fold_forest(pose)
    assert (separate.edges[0, :, 0] == EdgeType.root_jump).sum() == 2
    without_labels = pose.replace(pdb_info=None)
    with pytest.raises(ValueError, match="chain_id is required"):
        FoldForest.reasonable_fold_forest(without_labels)
    with pytest.raises(ValueError, match="shape"):
        FoldForest.reasonable_fold_forest(without_labels, chain_id=["A", "A"])
    with pytest.raises(ValueError, match="every real block"):
        FoldForest.reasonable_fold_forest(without_labels, chain_id=[[None, None]])
    empty = FoldForest.reasonable_fold_forest(pose[:0])
    assert empty.edges.shape == (0, 1, 4) and empty.n_edges.size == 0
    gap = FoldForest.reasonable_fold_forest(without_labels, chain_id=[["A", "A"]])
    assert (gap.edges[0, :, 0] == EdgeType.root_jump).sum() == 1
    np.testing.assert_array_equal(gap.edges[0, 1], [EdgeType.jump, 0, 1, 0])
