"""Changing parameter owners and charge annotations preserves rendered physics."""

import attrs
import numpy as np

from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES
from tmol_core.scoring.elec import partial_charge_table, partial_charges_for_block
from tmol_jax.score.score_function import ScoreFunction
from tmol_jax.score.score_types import ScoreType


def test_charge_scaling_owner_changes_and_mutable_overrides():
    b = Execution("jax-cpu").molecule(DOMAIN_CASES["mixed_batch"], terms=("fa_elec",))
    db, pose = b.parameter_database, b.pose

    def render(database):
        score = ScoreFunction(database)
        score.set_weight(ScoreType.fa_elec, 1)
        return score.jitted_value_and_grad(pose)

    original = render(db)
    expected = tuple(np.asarray(x) for x in original(pose.coords))
    assert all(np.linalg.norm(x) > 1e-3 for x in expected)
    changed = attrs.evolve(
        db,
        scoring=attrs.evolve(
            db.scoring,
            elec=attrs.evolve(
                db.scoring.elec,
                atom_charge_parameters=tuple(
                    attrs.evolve(row, charge=1.5 * row.charge)
                    for row in db.scoring.elec.atom_charge_parameters
                ),
            ),
        ),
    )
    altered = render(changed)

    def check(function, scale):
        for actual, reference in zip(function(pose.coords), expected):
            np.testing.assert_allclose(actual, scale * reference, rtol=3e-5, atol=3e-5)

    check(altered, 2.25)
    check(render(db), 1)
    table = partial_charge_table(db.scoring.elec)
    types = pose.packed_block_types.active_block_types
    previous = [getattr(bt, "elec_partial_charges", None) for bt in types]
    try:
        for bt, override in zip(types, previous):
            charges = partial_charges_for_block(bt, table, overrides=override)
            bt.elec_partial_charges = {
                atom.name: float(2 * charge) for atom, charge in zip(bt.atoms, charges)
            }
        doubled = render(db)
        check(doubled, 4)
        for bt in types:
            for atom in bt.elec_partial_charges:
                bt.elec_partial_charges[atom] *= 2
        check(render(db), 16)
        # Existing compiled snapshots retain their parameters after later renders.
        check(original, 1)
        check(altered, 2.25)
        check(doubled, 4)
    finally:
        for bt, override in zip(types, previous):
            if override is None:
                del bt.elec_partial_charges
            else:
                bt.elec_partial_charges = override
    check(render(db), 1)
