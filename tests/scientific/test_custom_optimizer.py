"""Molecular optimizer injection preserves physical movement and input identity."""

import numpy as np
import pytest

from .constraint_execution import Constraints
from .molecules import CASES


def gradient_step(objective, initial, *, lr, value_and_grad=None, **options):
    """Small functional optimizer used only by this execution contract test."""
    import jax

    if value_and_grad is None:

        def value_and_grad(x):
            return objective(x), jax.grad(lambda y: objective(y).sum())(x)

    values, gradient = value_and_grad(initial)
    result = initial - lr * gradient
    final, gradient = value_and_grad(result)
    return dict(x=result, grad=gradient, initial_fun=values.sum(), fun=final.sum())


@pytest.mark.parametrize("name", ("ligand", "connected_noncanonical", "mixed_batch"))
def test_custom_cartesian_optimizer_follows_analytic_molecular_target(backend, name):
    backend.molecule(CASES[name], terms=("constraint",))
    initial = backend.coords().copy()
    offsets = backend.numpy(backend.pose.block_coord_offset)
    atoms, parameters, movable = [], [], []
    mask = np.zeros(initial.shape[:2], bool)
    for pose, sequence in enumerate(CASES[name].sequences):
        block = len(sequence) - 1
        first = int(offsets[pose, block])
        for atom in (0, 1):
            target = initial[pose, first + atom] + np.array([2, 0, 0], np.float32)
            atoms.append([[pose, block, atom]])
            parameters.append([0, *target, 2])
        movable.append((pose, first))
        mask[pose, first] = True
    Constraints(backend).attach([("harmonic_coordinate", atoms, parameters)])
    before = backend.scores(initial).sum()
    if backend.framework == "jax":
        from tmol_jax import run_cart_min

        def optimizer(objective, initial, *, segment_ids, **settings):
            np.testing.assert_array_equal(
                backend.numpy(segment_ids), np.repeat(np.arange(len(movable)), 3)
            )
            return gradient_step(objective, initial, **settings)

        result = run_cart_min(
            backend.pose,
            backend.score_function,
            mask,
            optimizer=optimizer,
            optimizer_kwargs={"lr": 0.25},
        )
    else:
        from tmol import run_cart_min

        result = run_cart_min(
            backend.pose,
            backend.score_function,
            coord_mask=backend.module.as_tensor(mask, device=backend.device),
            optimizer_cls=backend.module.optim.SGD,
            optimizer_kwargs={"lr": 0.25},
        )
    actual = backend.numpy(result.coords)
    expected = initial.copy()
    for pose, atom in movable:
        # E=|x-target|^2/4: one lr=1/4 step moves 1/8 of the displacement.
        expected[pose, atom, 0] += 0.25
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(actual[~mask], initial[~mask])
    np.testing.assert_array_equal(backend.coords(), initial)
    assert actual.dtype == initial.dtype and result.device == backend.pose.device
    np.testing.assert_allclose(before, 2 * len(movable), atol=2e-5)
    np.testing.assert_allclose(
        backend.scores(actual).sum(), 1.765625 * len(movable), atol=2e-5
    )


@pytest.mark.parametrize("name", ("ligand", "d_peptide"))
def test_custom_kinematic_optimizer_uses_shared_movement_policy(backend, name):
    from .kinematics import MolecularKinematicExecution, MOLECULAR_KINEMATIC_TERMS
    from .molecules import DOMAIN_CASES

    backend.molecule(DOMAIN_CASES[name], terms=MOLECULAR_KINEMATIC_TERMS)
    initial = backend.coords().copy()
    execution = MolecularKinematicExecution(backend, initial)
    movement = execution.move_map()
    movement.move_all_named_torsions = True
    calls = []
    if backend.framework == "jax":
        from tmol_jax import run_kin_min
        from tmol_jax.optimization import armijo_minimize

        def optimizer(*args, **kwargs):
            calls.append(True)
            return armijo_minimize(*args, **kwargs)

        result = run_kin_min(
            backend.pose,
            backend.score_function,
            movement,
            optimizer=optimizer,
            optimizer_kwargs={"max_iter": 3},
        )
    else:
        from tmol import FoldForest, run_kin_min
        from tmol.optimization._lbfgs_armijo import LBFGS_Armijo

        class Optimizer(LBFGS_Armijo):
            def step(self, *args, **kwargs):
                calls.append(True)
                return super().step(*args, **kwargs)

        result = run_kin_min(
            backend.pose,
            backend.score_function,
            FoldForest.reasonable_fold_forest(backend.pose),
            movement,
            optimizer_cls=Optimizer,
            optimizer_kwargs={"max_iter": 3},
        )
    assert calls == [True]
    actual = backend.numpy(result.coords)
    assert np.isfinite(actual).all()
    assert backend.scores(actual).sum() < backend.scores(initial).sum() - 1e-3
    np.testing.assert_array_equal(backend.coords(), initial)
