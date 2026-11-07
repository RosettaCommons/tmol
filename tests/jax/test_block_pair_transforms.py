"""Fixed chemical topology permits transformed pair attribution after motion."""

import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import CASES


def test_block_pair_jit_vmap_motion_and_arbitrary_pair_cotangents():
    name = (
        "jax-cuda"
        if "jax-cuda" in os.environ.get("TMOL_TEST_BACKENDS", "")
        else "jax-cpu"
    )
    b = Execution(name).molecule(CASES["mixed_batch"], terms=())
    b.use_beta2016()
    with jax.enable_x64():
        pair = b.score_function.render_block_pair_scoring_module(b.pose)
        x = b.pose.coords.astype(jnp.float64)
        moved = x.at[0, int(b.pose.block_coord_offset[0, 2]) :].add(
            jnp.array([20.0, 0.0, 0.0])
        )
        samples = jnp.stack((x, moved))
        actual = jax.jit(jax.vmap(pair))(samples)
        whole = b.score_function.render_whole_pose_scoring_module(b.pose)
        expected = jax.jit(jax.vmap(whole))(samples)
        assert actual.device == b.device
        np.testing.assert_allclose(actual.sum((-2, -1)), expected, atol=2e-8, rtol=2e-8)
        assert np.max(np.abs(np.asarray(actual[1] - actual[0]))) > 1e-3
        weights = jnp.asarray(np.random.default_rng(718).normal(size=actual.shape[1:]))
        objective = jax.jit(lambda x: jnp.sum(pair(x) * weights))
        gradient = jax.jit(jax.grad(objective))(x)
        direction = jnp.asarray(np.random.default_rng(44).normal(size=x.shape))
        direction /= jnp.linalg.norm(direction)
        finite = (
            objective(x + 1e-5 * direction) - objective(x - 1e-5 * direction)
        ) / 2e-5
        np.testing.assert_allclose(
            jnp.sum(gradient * direction), finite, atol=2e-6, rtol=2e-6
        )


def test_selected_pairs_transform_dynamic_indices_and_invalid_bounds():
    name = (
        "jax-cuda"
        if "jax-cuda" in os.environ.get("TMOL_TEST_BACKENDS", "")
        else "jax-cpu"
    )
    b = Execution(name).molecule(CASES["mixed_batch"], terms=("fa_ljatr",))
    pair = b.score_function.render_block_pair_scoring_module(b.pose)
    x = b.pose.coords
    pairs = jax.device_put(np.array([[0, 0], [-1, -1]], np.int32), b.device)
    execute = jax.jit(jax.vmap(pair.score_interactions, in_axes=(0, None)))
    result = execute(jnp.stack((x, x + 0.5)), pairs)
    assert result.device == b.device
    np.testing.assert_allclose(result[0], result[1], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(result[0], pair.score_interactions(x, pairs))
    invalid = pairs.at[0, 0].set(b.pose.max_n_blocks)
    with pytest.raises(IndexError, match="out of range"):
        pair.score_interactions(x, invalid)
    assert np.isnan(np.asarray(execute(jnp.stack((x, x)), invalid))).all()
