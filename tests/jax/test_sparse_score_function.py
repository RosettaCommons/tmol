"""Sparse configuration snapshots, named channels and transform boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.rotamer_execution import RotamerExecution
from tests.scientific.test_complete_preset import CASES, CHANNELS
from tmol_jax import ScoreType


@pytest.mark.parametrize("name", CASES)
def test_complete_sparse_named_channels(name):
    b = Execution("jax-cpu").molecule(CASES[name], terms=("ref",))
    e = RotamerExecution(b, family="genbonded")
    b.use_beta2016()
    scorer = b.score_function.render_rotamer_scoring_module(
        b.pose, e.rotamers, sum_terms=False, apply_weights=False
    )
    assert {st.name for st in scorer.score_types} == CHANNELS
    values, indices = jax.jit(scorer.prepare(e.rotamers.coords))(e.rotamers.coords)
    totals = np.zeros((25, b.pose.n_poses))
    for c in range(25):
        np.add.at(totals[c], np.asarray(indices)[0], np.asarray(values)[c])
    np.testing.assert_allclose(
        totals,
        b.scores(b.coords(), sum_terms=False, apply_weights=False),
        rtol=3e-5,
        atol=3e-4,
    )


def test_sparse_configuration_refresh_snapshot_and_empty():
    b = Execution("jax-cpu").molecule(CASES["mixed_batch"], terms=("fa_elec", "ref"))
    e = RotamerExecution(b, family="elec")
    old = b.score_function.render_rotamer_scoring_module(b.pose, e.rotamers)
    fixed = old.prepare(e.rotamers.coords)
    values, indices = fixed(e.rotamers.coords)
    b.score_function.set_weight(ScoreType.fa_elec, 0)
    b.score_function.set_option("ref_weights", {"HYP": 7.0, "GLY": 2.0})
    new = b.score_function.render_rotamer_scoring_module(b.pose, e.rotamers)
    np.testing.assert_array_equal(fixed(e.rotamers.coords)[0], values)
    np.testing.assert_array_equal(old(e.rotamers.coords)[1], indices)
    assert not np.isclose(new(e.rotamers.coords)[0].sum(), values.sum())
    with pytest.raises(ValueError, match="outside JAX transformations"):
        jax.jit(old)(e.rotamers.coords)
    bv, bg = jax.jit(jax.vmap(jax.value_and_grad(lambda x: fixed(x)[0].sum())))(
        jnp.stack((e.rotamers.coords, e.rotamers.coords))
    )
    np.testing.assert_array_equal(bv[0], bv[1])
    np.testing.assert_array_equal(bg[0], bg[1])
    b.score_function.set_weight(ScoreType.ref, 0)
    empty = b.score_function.render_rotamer_scoring_module(b.pose, e.rotamers)
    assert empty(e.rotamers.coords)[0].shape == (0,)
