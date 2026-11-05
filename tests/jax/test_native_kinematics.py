"""Runtime transforms for the shared native coordinate-to-DOF operation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.kinematics._native import inverse_coordinates
from tmol_jax.kinematics import KinematicForest


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
