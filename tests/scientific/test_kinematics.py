"""Movable-DOF derivatives and rigid motion on the same branched native forest."""

from contextlib import nullcontext

import numpy as np
import pytest

from .kinematics import KinematicExecution, branched_forest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_branched_kinematics_movable_dof_derivatives(backend, dtype):
    parent, kinds, reference = branched_forest(dtype)
    context = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with context:
        execution = KinematicExecution(backend, parent, kinds, reference)
        initial = reference[:, :6].copy()
        weights = np.random.default_rng(29).normal(size=(6, 3)).astype(dtype)
        coords, gradient = execution.evaluate(initial, weights)
        assert coords.dtype == dtype and gradient.dtype == dtype
        assert np.isfinite(coords).all() and np.isfinite(gradient).all()
        np.testing.assert_array_equal(coords[0], 0)
        # Reference angles are fixed preparation data. Every exposed column is
        # tested, including unused bond/root columns whose derivatives are zero.
        precise = KinematicExecution(
            backend, parent, kinds, reference.astype(np.float64)
        )
        numeric = np.zeros_like(initial, dtype=np.float64)
        exact_initial = initial.astype(np.float64)
        step = 1e-5
        for index in np.ndindex(initial.shape):
            delta = np.zeros_like(exact_initial)
            delta[index] = step
            plus = precise.evaluate(exact_initial + delta)
            minus = precise.evaluate(exact_initial - delta)
            numeric[index] = ((plus - minus) * weights.astype(np.float64)).sum() / (
                2 * step
            )
        tolerance = 2e-5 if dtype == np.float32 else 2e-8
        np.testing.assert_allclose(gradient, numeric, rtol=tolerance, atol=tolerance)
        translated = initial.copy()
        translated[1, :3] += np.array([2, -1, 0.5], dtype)
        shifted = execution.evaluate(translated)
        np.testing.assert_allclose(
            shifted[1:] - coords[1:], np.tile([2, -1, 0.5], (5, 1)), atol=2e-6
        )
        _, zero = execution.evaluate(initial, np.zeros_like(weights))
        np.testing.assert_array_equal(zero, 0)

    if backend.framework == "jax":
        _check_inverse_transforms(backend, dtype)
        if dtype == np.float32:
            _check_bound_forest_contracts(backend)


def _check_inverse_transforms(backend, dtype):
    jax, jnp = backend.module, backend.module.numpy
    from tmol_jax.kinematics import KinematicForest

    with jax.default_device(backend.device):
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


def _check_bound_forest_contracts(backend):
    jax, jnp = backend.module, backend.module.numpy
    from tmol_jax.kinematics import KinematicForest
    from tmol_jax.kinematics._native import inverse_coordinates

    with jax.default_device(backend.device):
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

        from tests.scientific.kinematics import branched_forest
        from tmol_core.kinematics.scan_ordering import prepare_scan_ordering

        parent, kinds, reference = branched_forest(np.float32)
        zeros = np.zeros(len(parent), np.int32)
        forest = KinematicForest(
            parent,
            zeros,
            zeros,
            zeros,
            kinds,
            scan_ordering=prepare_scan_ordering(parent),
        )
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

        forest = KinematicForest(*(np.zeros(0, np.int32),) * 5)
        mapping = forest.bind(np.zeros((0, 9), np.float32))
        result = jax.jit(mapping)(mapping.initial_dofs)
        assert result.shape == (0, 3)
        gradient = jax.jit(jax.grad(lambda x: mapping(x).sum()))(mapping.initial_dofs)
        assert gradient.shape == (0, 6)
