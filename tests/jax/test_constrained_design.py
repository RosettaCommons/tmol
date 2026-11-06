"""Constrained design preserves differentiable coordinates and rejects lost identity."""

import os

import jax
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.constraint_execution import Constraints
from tests.scientific.test_constrained_design import constrained_design_fixture
from tmol_jax.pack.imposition import prepare_pose_imposition


def test_constrained_design_source_gradient_and_changed_atom_guard():
    name = (
        "jax-cuda"
        if "jax-cuda" in os.environ.get("TMOL_TEST_BACKENDS", "")
        else "jax-cpu"
    )
    b = Execution(name)
    rotamers, graph, choices, selected, _ = constrained_design_fixture(b)
    plan = prepare_pose_imposition(b.pose, rotamers, graph["retained"], choices)
    b.pose = plan.apply(rotamers.coords)
    b.use_beta2016()
    from tmol_jax import ScoreType

    b.score_function.set_weight(ScoreType.constraint, 1.0)
    score = b.score_function.render_whole_pose_scoring_module(b.pose)

    def objective(x):
        return score(plan.coordinates(x)).sum()
    value, gradient = jax.jit(jax.value_and_grad(objective))(rotamers.coords)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 1
    # Translation leaves every physical term and the distance constraints invariant.
    np.testing.assert_allclose(np.asarray(gradient).sum(0), 0, atol=3e-4)
    rng = np.random.default_rng(811)
    direction = rng.normal(size=rotamers.coords.shape).astype(np.float32)
    h = 1e-3
    numerical = (
        objective(rotamers.coords + h * direction)
        - objective(rotamers.coords - h * direction)
    ) / (2 * h)
    np.testing.assert_allclose(
        numerical, (np.asarray(gradient) * direction).sum(), rtol=4e-3, atol=4e-3
    )
    b = Execution(name)
    rotamers, graph, choices, selected, _ = constrained_design_fixture(b)
    blocks = b.pose.packed_block_types.active_block_types
    old = blocks[int(b.pose.block_type_ind[0, 0])]
    new = blocks[int(rotamers.block_type_ind_for_rot[selected[0, 0]])]
    lost = next(
        i
        for i, atom in enumerate(old.atoms)
        if i >= new.n_atoms or atom.name != new.atoms[i].name
    )
    Constraints(b).attach([("harmonic", [[[0, 0, lost], [0, 1, 0]]], [[2.3, 0.4]])])
    with pytest.raises(NotImplementedError, match="changed constraint atom identities"):
        prepare_pose_imposition(b.pose, rotamers, graph["retained"], choices)
