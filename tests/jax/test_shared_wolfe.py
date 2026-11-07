"""Compiled history storage and molecular masks around shared Wolfe policy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.optimization import lbfgs_minimize, run_cart_min
from tmol_jax.optimization._strong_wolfe_execution import JaxLBFGSExecution
from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES
from tests.scientific.test_molecular_minimization import _check_frozen_atom_minimization


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_fixed_history_matches_dense_inverse_hessian_after_wrap(dtype):
    # Independently form inverse BFGS matrices, rather than another two-loop
    # implementation. Exercise empty, partial, full, wrapped and reused histories.
    rng = np.random.default_rng(731)
    size, capacity = 7, 3
    gradient = jnp.asarray(rng.normal(size=size), dtype=dtype)
    execution = JaxLBFGSExecution(gradient, capacity)
    steps, differences, rho = [], [], []
    for iteration in range(8):
        if iteration:
            step = jnp.asarray(rng.normal(size=size), dtype=dtype)
            difference = step * jnp.asarray(np.arange(1, size + 1), dtype=dtype)
            steps = (steps + [step])[-capacity:]
            differences = (differences + [difference])[-capacity:]
            rho = (rho + [1 / jnp.dot(step, difference)])[-capacity:]
        scale = 0.3
        inverse = np.eye(size) * scale
        identity = np.eye(size)
        for step, difference in zip(steps, differences):
            s, y = np.asarray(step, dtype=np.float64), np.asarray(
                difference, dtype=np.float64
            )
            curvature = 1 / np.dot(s, y)
            transform = identity - curvature * np.outer(s, y)
            inverse = transform @ inverse @ transform.T + curvature * np.outer(s, s)
        expected = -inverse @ np.asarray(gradient)
        for _ in range(2):
            actual = execution.two_loop(gradient, steps, differences, rho, scale)
            assert actual.device == gradient.device and actual.dtype == gradient.dtype
            np.testing.assert_allclose(
                actual, expected, rtol=3e-6, atol=3e-7 if dtype == np.float32 else 2e-14
            )
        assert execution.steps.shape == (capacity, size)
        assert execution.count == min(iteration, capacity)


class _WolfeMolecule(Execution):
    def minimize(self, move_mask, iterations=5):
        with jax.default_device(self.device):
            pose = run_cart_min(
                self.pose,
                self.score_function,
                move_mask,
                max_iter=iterations,
                optimizer="strong_wolfe",
            )
        assert pose.coords.device == self.device
        return self.numpy(pose.coords)


@pytest.mark.parametrize(
    "name",
    (
        "ligand",
        "connected_noncanonical",
        "d_peptide",
        "modified_dna",
        "metal_mixed",
        "mixed_batch",
    ),
)
def test_full_molecular_wolfe_uses_shared_fixed_atom_contract(name):
    device = jax.devices()[0]
    backend = _WolfeMolecule("jax-" + device.platform)
    backend.molecule(DOMAIN_CASES[name], terms=())
    backend.use_beta2016()
    _check_frozen_atom_minimization(backend)


def test_invalid_trial_does_not_report_convergence():
    # A finite initial point with no finite improving trial must remain in place,
    # and exhaustion of the line search must not be reported as convergence.
    def objective(x):
        return jnp.where(x[0] == 0, 1.0, jnp.inf), jnp.ones_like(x)

    result = lbfgs_minimize(
        None, jnp.zeros(1), max_iter=3, line_search_max_iter=2, value_and_grad=objective
    )
    np.testing.assert_array_equal(result["x"], [0])
    assert np.isfinite(result["fun"])
    assert not result["converged"]


@pytest.mark.parametrize("name", ("ligand", "connected_noncanonical", "d_peptide"))
def test_kinematic_wolfe_retains_shared_molecular_minimization_contract(
    monkeypatch, name
):
    from tmol_jax import run_kin_min
    from tests.scientific.kinematics import MolecularKinematicExecution
    from tests.scientific.test_molecular_movement import (
        test_internal_minimization_decreases_real_molecular_energy as check,
    )

    def minimize(execution, move_map, iterations):
        backend = execution.backend
        with jax.default_device(backend.device):
            pose = run_kin_min(
                backend.pose,
                backend.score_function,
                move_map,
                max_iter=iterations,
                optimizer="strong_wolfe",
            )
        return backend.numpy(pose.coords)

    monkeypatch.setattr(MolecularKinematicExecution, "minimize", minimize)
    check(Execution("jax-" + jax.devices()[0].platform), name)
