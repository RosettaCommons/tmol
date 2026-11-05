"""A weighted native reverse pass must equal independently exposed channels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES
from tests.scientific.test_lk_ball import TERMS
from tmol_jax import ScoreType


@pytest.mark.parametrize("weights", [(0.2, -0.7, 0, 0.5), (0, 0, 0, -1.3)])
def test_weighted_lk_ball_preserves_channel_selection_and_gradients(weights):
    backend = Execution("jax-" + ("cuda" if jax.default_backend() == "gpu" else "cpu"))
    backend.molecule(DOMAIN_CASES["connected_noncanonical"], terms=TERMS)
    channels = backend.score_function.render_whole_pose_scoring_module(
        backend.pose, sum_terms=False, apply_weights=False
    )
    for name, weight in zip(TERMS, weights):
        backend.score_function.set_weight(ScoreType[name], weight)
    weighted = backend.score_function.render_whole_pose_scoring_module(backend.pose)
    unweighted = backend.score_function.render_whole_pose_scoring_module(
        backend.pose, apply_weights=False
    )
    x = backend.pose.coords
    for function, coefficients in (
        (weighted, weights),
        (unweighted, np.not_equal(weights, 0)),
    ):

        def reference(x):
            return (jnp.asarray(coefficients, x.dtype)[:, None] * channels(x)).sum()

        actual = jax.jit(jax.value_and_grad(lambda x: function(x).sum()))(x)
        expected = jax.jit(jax.value_and_grad(reference))(x)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=3e-6, atol=3e-6)
