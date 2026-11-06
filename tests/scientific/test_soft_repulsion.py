"""The shared soft-repulsive option changes only repulsive LJ physics."""

from contextlib import nullcontext
import numpy as np
import pytest
from .molecules import DOMAIN_CASES


@pytest.mark.parametrize(
    "name", ("ligand", "connected_noncanonical", "d_peptide", "mixed_batch")
)
def test_soft_repulsion_matches_whole_pair_and_independent_derivatives(backend, name):
    b = backend.molecule(
        DOMAIN_CASES[name], terms=("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")
    )
    with b.module.enable_x64() if b.framework == "jax" else nullcontext():
        x = b.coords().astype(np.float64)
        old = b.score_function.render_whole_pose_scoring_module(b.pose)
        hard, hard_g = b.channel_values_and_gradients(x)
        b.set_options({"soft_rep": True})
        soft, soft_g = b.channel_values_and_gradients(x)
        rep = b.channel_names().index("fa_ljrep")
        others = [i for i in range(len(hard)) if i != rep]
        np.testing.assert_array_equal(soft[others], hard[others])
        np.testing.assert_array_equal(soft_g[others], hard_g[others])
        assert np.max(hard[rep] - soft[rep]) > 1e-3
        assert np.all(soft[rep] <= hard[rep] + 1e-10)
        direction = np.random.default_rng(55).normal(size=x.shape)
        direction /= np.linalg.norm(direction)
        numeric = b.float64_directional_derivative(x, direction, step=1e-5)
        np.testing.assert_allclose(
            (soft_g * direction).sum((2, 3)), numeric, rtol=2e-6, atol=2e-6
        )
        value, gradient = b.value_and_grad(x)
        np.testing.assert_allclose(value, soft.sum(0), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(gradient, soft_g.sum(0), rtol=2e-10, atol=2e-10)
        pair = b.score_function.render_block_pair_scoring_module(b.pose)
        if b.framework == "jax":
            coords = b.module.device_put(x, b.device)
            matrix = b.module.jit(
                lambda x: pair(x, sum_terms=False, apply_weights=False)
            )(coords)
            old_value = b.module.jit(old)(coords)
        else:
            coords = b.module.as_tensor(x, device=b.device)
            matrix = pair(coords, sum_terms=False, apply_weights=False)
            old_value = old(coords)
        np.testing.assert_allclose(
            b.numpy(matrix).sum((-2, -1)), soft, rtol=2e-10, atol=2e-10
        )
        np.testing.assert_allclose(
            b.numpy(old_value), hard.sum(0), rtol=2e-10, atol=2e-10
        )
