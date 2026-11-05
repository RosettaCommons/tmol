"""Parameter context changes must affect real molecular physics or reject."""

import attrs
import jax
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import CASES
from tmol_jax import ScoreFunction, ScoreType


def test_ligand_charge_scaling_and_missing_lj_parameter_are_explicit():
    backend = Execution("jax-cpu").molecule(CASES["ligand"], terms=("fa_elec",))
    pose, db = backend.pose, backend.parameter_database
    old = backend.score_function.render_whole_pose_scoring_module(pose)
    rows = tuple(
        attrs.evolve(row, charge=2 * row.charge)
        for row in db.scoring.elec.atom_charge_parameters
    )
    modified = attrs.evolve(
        db,
        scoring=attrs.evolve(
            db.scoring, elec=attrs.evolve(db.scoring.elec, atom_charge_parameters=rows)
        ),
    )
    new = ScoreFunction(modified)
    new.set_weight(ScoreType.fa_elec, 1)
    score = new.render_whole_pose_scoring_module(pose)
    value, gradient = jax.value_and_grad(lambda x: old(x).sum())(pose.coords)
    changed, new_gradient = jax.value_and_grad(lambda x: score(x).sum())(pose.coords)
    assert abs(value) > 1e-3 and np.linalg.norm(gradient) > 1e-3
    np.testing.assert_allclose(changed, 4 * value, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(new_gradient, 4 * gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(old(pose.coords).sum(), value, rtol=3e-6, atol=3e-6)

    missing_type = pose.block_type(0, 0).atoms[0].atom_type
    incomplete = attrs.evolve(
        db,
        scoring=attrs.evolve(
            db.scoring,
            ljlk=attrs.evolve(
                db.scoring.ljlk,
                atom_type_parameters=tuple(
                    p
                    for p in db.scoring.ljlk.atom_type_parameters
                    if p.name != missing_type
                ),
            ),
        ),
    )
    sf = ScoreFunction(incomplete)
    sf.set_weight(ScoreType.fa_ljrep, 1)
    with pytest.raises(ValueError, match="missing or non-finite LJLK parameters"):
        sf.render_whole_pose_scoring_module(pose)
