"""Rendered reduction options retain molecular parameters after configuration edits."""

from contextlib import nullcontext
from importlib import import_module

import numpy as np
import pytest

from .molecules import CASES
from .test_selected_pair_scoring import _array


@pytest.mark.parametrize("name", ("ligand", "connected_noncanonical", "mixed_batch"))
def test_rendered_score_options_keep_channel_values_and_gradients(backend, name):
    b = backend.molecule(CASES[name], terms=())
    _, weights = b.use_beta2016()
    with b.module.enable_x64() if b.framework == "jax" else nullcontext():
        coords = b.coords().astype(np.float64)
        raw, channel_gradients = b.channel_values_and_gradients(coords)
        scorer = b.score_function.render_whole_pose_scoring_module(b.pose)
        # First requesting an alternate reduction after mutation must still
        # use the weights and chemistry of the original rendered snapshot.
        frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
        b.score_function.set_weight(frontend.ScoreType.fa_ljatr, -3.0)
        x = _array(b, coords)
        np.testing.assert_allclose(
            b.numpy(scorer.unweighted_scores(x)), raw, rtol=2e-10, atol=2e-10
        )
        for weighted in (False, True):
            expected = raw * weights[:, None] if weighted else raw
            for summed in (False, True):
                actual = scorer(x, sum_terms=summed, apply_weights=weighted)
                np.testing.assert_allclose(
                    b.numpy(actual),
                    expected.sum(0) if summed else expected,
                    rtol=2e-10,
                    atol=2e-10,
                )

        def objective(x):
            return scorer(x, sum_terms=False, apply_weights=False).sum()

        if b.framework == "jax":
            actual_gradient = b.module.jit(b.module.grad(objective))(x)
        else:
            x = x.detach().requires_grad_(True)
            actual_gradient = b.module.autograd.grad(objective(x), x)[0]
        np.testing.assert_allclose(
            b.numpy(actual_gradient), channel_gradients.sum(0), rtol=2e-10, atol=2e-10
        )
        assert np.linalg.norm(b.numpy(actual_gradient)) > 1e-3

    if b.framework == "jax" and name == "mixed_batch":
        _check_transformed_reductions(b)


def _check_transformed_reductions(b):
    import jax
    import jax.numpy as jnp

    b.molecule(CASES["mixed_batch"], terms=("fa_ljatr",))
    score = b.score_function.render_whole_pose_scoring_module(
        b.pose, sum_terms=False, apply_weights=False
    )
    x = b.pose.coords
    samples = jnp.stack((x, x + 0.25))
    raw = jax.jit(jax.vmap(score))(samples)
    total = jax.jit(jax.vmap(lambda x: score(x, sum_terms=True)))(samples)
    np.testing.assert_allclose(total, raw.sum(1), rtol=2e-6, atol=2e-6)
    actual, expected = np.asarray(score(x)), np.asarray(score.unweighted_scores(x))
    if b.device.platform == "gpu":
        # Repeated calls to the unchanged float32 CUDA scorer also vary under
        # atomic accumulation. Retain the score-lifecycle repeat bound; the
        # shared per-channel float64 option/gradient checks remain at 2e-10.
        bound = 2 * np.finfo(expected.dtype).eps * np.max(np.abs(expected))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=bound)
    else:
        np.testing.assert_array_equal(actual, expected)
    weighted = jax.jit(lambda x: score(x, sum_terms=True, apply_weights=True))(x)
    np.testing.assert_allclose(weighted, raw[0, 0], rtol=2e-6, atol=2e-6)
