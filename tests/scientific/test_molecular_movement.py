"""Shared chemical movement priorities and internal minimization invariants."""

from contextlib import nullcontext

import numpy as np
import pytest

from .kinematics import MolecularKinematicExecution, MOLECULAR_KINEMATIC_TERMS
from .molecules import DOMAIN_CASES


@pytest.mark.parametrize("name", DOMAIN_CASES)
def test_molecular_movement_defaults_and_explicit_overrides(backend, name):
    terms = (
        MOLECULAR_KINEMATIC_TERMS
        if backend.framework == "jax" and name == "mixed_batch"
        else ("fa_ljrep",)
    )
    backend.molecule(DOMAIN_CASES[name], terms=terms)
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

    if backend.framework == "jax" and name == "mixed_batch":
        import jax
        from tmol_jax import MoveMap, PoseStackKinematics, run_kin_min

        with jax.enable_x64(True), jax.default_device(backend.device):
            for dtype in (np.float32, np.float64):
                pose = backend.pose.replace(coords=backend.pose.coords.astype(dtype))
                move_map = MoveMap.from_pose_stack(pose)
                module = PoseStackKinematics(pose)
                mapping = module.bind()
                unchanged, result = run_kin_min(
                    pose, backend.score_function, move_map, return_result=True
                )
                np.testing.assert_array_equal(unchanged.coords, pose.coords)
                assert result["n_iter"] == 0 and np.asarray(result["converged"]).all()
                move_map.move_all_named_torsions = True
                mask = np.asarray(module.resolve_move_map(move_map)).reshape(-1)
                minimized, result = run_kin_min(
                    pose,
                    backend.score_function,
                    move_map,
                    max_iter=3,
                    return_result=True,
                )
                assert minimized.coords.dtype == dtype
                assert result["fun"] < result["initial_fun"]
                np.testing.assert_array_equal(
                    np.asarray(result["x"])[~mask],
                    np.asarray(mapping.initial_dofs).reshape(-1)[~mask],
                )
                np.testing.assert_array_equal(np.asarray(result["grad"])[~mask], 0)
                move_map.move_all_named_torsions = False
                assert mask.any() and not module.resolve_move_map(move_map).any()
                wrong = MoveMap(1, 1, 0, 1)
                with pytest.raises(ValueError, match="dimensions"):
                    module.resolve_move_map(wrong)


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
