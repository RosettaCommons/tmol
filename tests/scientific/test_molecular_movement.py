"""Shared chemical movement priorities and internal minimization invariants."""

from contextlib import nullcontext

import numpy as np
import pytest

from .kinematics import MolecularKinematicExecution, MOLECULAR_KINEMATIC_TERMS
from .molecules import DOMAIN_CASES


@pytest.mark.parametrize("name", DOMAIN_CASES)
def test_molecular_movement_defaults_and_explicit_overrides(backend, name):
    backend.molecule(DOMAIN_CASES[name], terms=("fa_ljrep",))
    execution = MolecularKinematicExecution(backend, backend.coords())
    move_map = execution.move_map()
    assert not execution.movement_mask(move_map).any()
    move_map.move_all_root_jumps = True
    mask = execution.movement_mask(move_map)
    assert mask[:, :3].any() and not mask[0].any()
    move_map.set_move_all_jump_dofs_for_root_jump(0, value=False)
    disabled = execution.movement_mask(move_map)
    atoms_in_first_pose = execution.atom_ids // backend.coords().shape[1] == 0
    assert not disabled[atoms_in_first_pose].any()
    np.testing.assert_array_equal(
        disabled[~atoms_in_first_pose], mask[~atoms_in_first_pose]
    )
    move_map.move_all_named_torsions = True
    named = execution.movement_mask(move_map)
    # Named torsions can rotate bonds; they never enable bond lengths or angles.
    newly_moving = named & ~disabled
    assert not newly_moving[:, 1:3].any()
    if name not in ("metal_cluster",):
        assert newly_moving.any()


@pytest.mark.parametrize("name", ("ligand", "connected_noncanonical", "d_peptide"))
def test_internal_minimization_decreases_real_molecular_energy(backend, name):
    backend.molecule(DOMAIN_CASES[name], terms=MOLECULAR_KINEMATIC_TERMS)
    initial = backend.coords().copy()
    execution = MolecularKinematicExecution(backend, initial)
    move_map = execution.move_map()
    move_map.move_all_named_torsions = True
    before = backend.scores(initial).sum()
    for optimizer in (
        ("armijo", "strong_wolfe") if backend.framework == "jax" else ("armijo",)
    ):
        result = execution.minimize(move_map, 5, optimizer=optimizer)
        after = backend.scores(result).sum()
        assert np.isfinite(result).all() and np.isfinite(after)
        assert after < before - 1e-3
        assert np.linalg.norm(result - initial) > 1e-4
        np.testing.assert_array_equal(backend.coords(), initial)


@pytest.mark.parametrize("name", ("ligand", "d_peptide"))
def test_double_kinematics_preserves_float_coordinate_storage(backend, name):
    context = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with context:
        backend.molecule(DOMAIN_CASES[name], terms=MOLECULAR_KINEMATIC_TERMS)
        initial = backend.coords().copy()
        assert initial.dtype == np.float32
        execution = MolecularKinematicExecution(backend, initial, kin_dtype=np.float64)
        dofs = execution.initial_dofs
        assert dofs.dtype == np.float64
        rebuilt = execution.evaluate(dofs)
        assert rebuilt.dtype == initial.dtype
        np.testing.assert_allclose(rebuilt, initial, rtol=2e-6, atol=2e-6)
        weights = (
            np.random.default_rng(728).normal(size=initial.shape).astype(np.float32)
        )
        _, gradient = execution.evaluate(dofs, weights)
        assert gradient.dtype == np.float64 and np.isfinite(gradient).all()
        direction = np.random.default_rng(275).normal(size=dofs.shape)
        direction /= np.linalg.norm(direction)
        step = 2e-3
        numeric = (
            (
                execution.evaluate(dofs + step * direction).astype(float)
                - execution.evaluate(dofs - step * direction).astype(float)
            )
            * weights
        ).sum() / (2 * step)
        np.testing.assert_allclose(
            (gradient * direction).sum(), numeric, rtol=2e-3, atol=2e-3
        )
        movement = execution.move_map()
        movement.move_all_named_torsions = True
        minimized = execution.minimize(movement, 5)
        assert minimized.dtype == initial.dtype
        if backend.framework == "jax":
            diagnostics = execution.minimization_diagnostics
            assert diagnostics["x"].dtype == diagnostics["grad"].dtype == np.float64
        assert backend.scores(minimized).sum() < backend.scores(initial).sum() - 1e-3
        np.testing.assert_array_equal(backend.coords(), initial)
