"""Independent objective and inverse-Hessian contracts for shared optimization."""

from contextlib import nullcontext

import numpy as np
import pytest


def _services(backend, dtype):
    if backend.framework == "jax":
        from tmol_jax.optimization._armijo_native import ArmijoExecution, finalize
        from tmol_jax.optimization._compact_lbfgs import compact_direction

        context = backend.module.enable_x64(True)

        def array(value, kind=dtype):
            return backend.module.device_put(np.asarray(value, kind), backend.device)

        return context, array, ArmijoExecution, finalize, compact_direction
    from tmol.optimization._lbfgs_armijo import (
        _TorchArmijoExecution,
        _lbfgs_two_loop_impl,
    )
    from tmol.optimization._armijo_compiled import _armijo_finalize

    def array(value, kind=dtype):
        return backend.module.as_tensor(np.asarray(value, kind), device=backend.device)

    return (
        nullcontext(),
        array,
        _TorchArmijoExecution,
        _armijo_finalize,
        _lbfgs_two_loop_impl,
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_segmented_armijo_acceptance_inactive_and_failed_objectives(backend, dtype):
    from tmol_core.optimization.armijo import armijo_linesearch_segmented

    context, array, execution, finalize, _ = _services(backend, dtype)
    with context:
        initial = array([1, 0.1, 4, 7, 4e-8])
        searching = array([True, True, True, False, True], bool)
        derivative, phi0 = array([-4] * 5), array([2] * 5)

        def objective(alpha):
            value = 0.5 * (2 - 2 * alpha) ** 2
            if backend.framework == "jax":
                return value.at[4].set(np.nan)
            value[4] = np.nan
            return value

        accepted, values, evaluations, status, _ = armijo_linesearch_segmented(
            objective,
            derivative,
            phi0,
            initial,
            searching,
            minstep=1e-6,
            execution=execution,
            combine_checks=True,
        )
        step, failed = finalize(status, derivative, accepted, initial, searching, 1e-6)
        atol = 1e-6 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(
            backend.numpy(accepted), [1, 0.2, 1, 0, 0], atol=atol, rtol=0
        )
        np.testing.assert_allclose(
            backend.numpy(values), [0, 1.28, 0, 2, 2], atol=atol, rtol=0
        )
        np.testing.assert_array_equal(backend.numpy(status), [0, 0, 0, 0, 3])
        np.testing.assert_allclose(
            backend.numpy(step), [1, 0.2, 1, 7, 0.5], atol=atol, rtol=0
        )
        np.testing.assert_array_equal(backend.numpy(failed), [False] * 4 + [True])
        assert evaluations == 2


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_compact_direction_matches_dense_bfgs_with_wrapped_partial_history(
    backend, dtype
):
    context, array, _, _, direction = _services(backend, dtype)
    random = np.random.default_rng(74)
    steps = random.normal(size=(5, 2, 6)).astype(dtype)
    differences = steps * np.arange(1, 7, dtype=dtype)[None, None, :]
    steps[1, 0], differences[1, 0] = 0, 0
    gradient = random.normal(size=(2, 6)).astype(dtype)
    expected = []
    for pose in range(2):
        inverse = np.eye(6)
        for index in [3, 4, 0, 1, 2]:
            s, y = steps[index, pose].astype(float), differences[index, pose].astype(
                float
            )
            if not np.any(s):
                continue
            rho = 1 / np.dot(y, s)
            left = np.eye(6) - rho * np.outer(s, y)
            inverse = left @ inverse @ left.T + rho * np.outer(s, s)
        expected.append(-inverse @ gradient[pose])
    with context:
        g = array(gradient)
        actual = direction(g, array(differences), array(steps), history_start=3)
    assert actual.dtype == g.dtype
    np.testing.assert_allclose(
        backend.numpy(actual),
        expected,
        atol=2e-6 if dtype == np.float32 else 2e-12,
        rtol=2e-6 if dtype == np.float32 else 2e-12,
    )
