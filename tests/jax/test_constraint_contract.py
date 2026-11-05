"""Reject invalid constraint inputs before coordinates reach JAX gathers."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.score.constraint.constraint_energy_term import (
    ConstraintEnergyTerm,
    ConstraintSet,
)
from tmol_jax.score.score_types import ScoreType


def pose():
    return SimpleNamespace(coords=jnp.zeros((1, 2, 3)))


def constraint(indices):
    return ConstraintSet(
        target_coords=jnp.zeros((1, 3)),
        atom_indices=indices,
        k=jnp.ones((1,)),
        tol=jnp.zeros((1,)),
    )


@pytest.mark.parametrize("attached", [False, True])
def test_unsupported_constraints_are_not_silently_ignored(attached):
    p = pose()
    options = {}
    if attached:
        p.constraint_set = object()
    else:
        options["constraint_set"] = object()
    with pytest.raises(TypeError, match="ConstraintSet"):
        ConstraintEnergyTerm().setup_for_pose_stack(p, **options)


@pytest.mark.parametrize(
    "indices",
    [np.array([[0.0, 1.5]], dtype=np.float32), np.array([[False, True]])],
)
def test_constraint_atom_indices_are_not_coerced(indices):
    with pytest.raises(TypeError, match="integer"):
        ConstraintEnergyTerm().setup_for_pose_stack(
            pose(), constraint_set=constraint(indices)
        )


def test_valid_constraint_value_and_gradient_preserved():
    p = pose()
    term = ConstraintEnergyTerm()
    state = term.setup_for_pose_stack(
        p, constraint_set=constraint(jnp.array([[0, 1]], dtype=jnp.int32))
    )
    coordinates = p.coords.at[0, 1, 0].set(2.0)
    evaluate = jax.jit(
        jax.value_and_grad(lambda x: term(p, state, x)[ScoreType.constraint].sum())
    )
    value, gradient = evaluate(coordinates)
    np.testing.assert_allclose(value, 2.0)
    expected = np.zeros((1, 2, 3), dtype=np.float32)
    expected[0, 1, 0] = 2.0
    np.testing.assert_allclose(gradient, expected)


def test_absent_constraints_remain_zero():
    p = pose()
    term = ConstraintEnergyTerm()
    state = term.setup_for_pose_stack(p)
    np.testing.assert_array_equal(term(p, state)[ScoreType.constraint], [0.0])
