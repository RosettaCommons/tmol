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
