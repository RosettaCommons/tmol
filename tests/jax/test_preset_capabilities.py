"""Complete declared presets and explicit boundaries for active chemistry."""

import jax
import numpy as np
import pytest
import yaml

from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES
from tmol_jax import ScoreFunction, ScoreType, beta2016_score_function
from tmol_jax.database import default_directory


def test_preset_keeps_every_declared_weight():
    expected = yaml.safe_load(
        (default_directory() / "score_functions/beta2016.sfxn").read_text()
    )["weights"]
    score = beta2016_score_function()
    assert {st.name: weight for st, weight in score.weights.items()} == expected
    assert ScoreType.na_torsion in score.active_score_types()
    assert ScoreType.metal_coordination in score.active_score_types()


@pytest.mark.parametrize(
    "name,term",
    [
        ("dna", "na_torsion"),
        ("rna", "na_torsion_well"),
        ("metal_mixed", "metal_coordination"),
    ],
)
def test_unsupported_active_science_rejects_before_tracing(name, term):
    backend = Execution("jax-cpu").molecule(DOMAIN_CASES[name], terms=())
    sf = ScoreFunction(backend.parameter_database)
    sf.set_weight(ScoreType[term], 1)
    with pytest.raises(NotImplementedError, match=term):
        sf.render_whole_pose_scoring_module(backend.pose)
    sf.set_weight(ScoreType[term], 0)
    score = sf.render_whole_pose_scoring_module(backend.pose)
    np.testing.assert_array_equal(jax.jit(score)(backend.pose.coords), 0)


def test_ineligible_terms_preserve_empty_channels_and_coordinate_gradients():
    # A parameterized beta-amino-acid pair is neither NA nor metal; applicability
    # comes from shared chemistry, rather than canonical residue-name filtering.
    backend = Execution("jax-cpu").molecule(
        DOMAIN_CASES["connected_noncanonical"], terms=()
    )
    sf = ScoreFunction(backend.parameter_database)
    for st in (
        ScoreType.na_torsion,
        ScoreType.na_torsion_well,
        ScoreType.metal_coordination,
    ):
        sf.set_weight(st, 1)
    score = sf.render_whole_pose_scoring_module(backend.pose, sum_terms=False)
    assert score(backend.pose.coords).shape == (3, 1)
    np.testing.assert_array_equal(jax.jit(score)(backend.pose.coords), 0)
    np.testing.assert_array_equal(
        jax.jit(jax.grad(lambda x: score(x).sum()))(backend.pose.coords), 0
    )
