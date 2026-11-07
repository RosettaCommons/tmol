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
