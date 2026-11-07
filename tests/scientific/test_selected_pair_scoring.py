"""Public pair selection conserves molecular energy and physical derivatives."""

from contextlib import nullcontext

import numpy as np
import pytest

from .test_block_pair_scoring import CASES


def _array(backend, value):
    if backend.framework == "jax":
        return backend.module.device_put(value, backend.device)
    return backend.module.as_tensor(value, device=backend.device)


@pytest.mark.parametrize("name", ("ligand", "connected_noncanonical", "mixed_batch"))
def test_selected_pair_partition_and_physical_gradient(backend, name):
    b = backend.molecule(CASES[name], terms=())
    _, weights = b.use_beta2016()
    with b.module.enable_x64() if b.framework == "jax" else nullcontext():
        pair = b.score_function.render_block_pair_scoring_module(b.pose)
        x = _array(b, b.coords().astype(np.float64))
        count = b.pose.max_n_blocks
        all_pairs = np.array(
            [(i, j) for i in range(count) for j in range(count)], np.int32
        )
        left, right = (_array(b, part) for part in (all_pairs[::2], all_pairs[1::2]))
        # Disjoint selections reconstruct independent whole-pose energies,
        # including zero-weight channels and asymmetric native pair ownership.
        whole = b.scores(b.numpy(x), sum_terms=False, apply_weights=False)
        raw = pair.unweighted_scores(x)
        np.testing.assert_allclose(
            b.numpy(raw).sum((-2, -1)), whole, rtol=2e-8, atol=2e-8
        )
        for weighted in (False, True):
            selected = pair.score_interactions(
                x, left, sum_terms=False, apply_weights=weighted
            ) + pair.score_interactions(
                x, right, sum_terms=False, apply_weights=weighted
            )
            expected = whole * weights[:, None] if weighted else whole
            np.testing.assert_allclose(
                b.numpy(selected), expected, rtol=2e-8, atol=2e-8
            )
        assert np.linalg.norm(whole) > 1e-3
        empty = _array(b, np.empty((0, 2), np.int32))
        np.testing.assert_array_equal(b.numpy(pair.score_interactions(x, empty)), 0)
        doubled = _array(b, np.concatenate((all_pairs[::2], all_pairs[::2])))
        np.testing.assert_allclose(
            b.numpy(pair.score_interactions(x, doubled)),
            2 * b.numpy(pair.score_interactions(x, left)),
            rtol=2e-12,
            atol=2e-12,
        )

        def objective(coords):
            return pair.score_interactions(coords, left).sum()

        if b.framework == "jax":
            gradient = b.module.jit(b.module.grad(objective))(x)
        else:
            x = x.detach().requires_grad_(True)
            gradient = b.module.autograd.grad(objective(x), x)[0]
        assert np.linalg.norm(b.numpy(gradient)) > 1e-3
        direction = np.random.default_rng(241).normal(size=x.shape)
        direction = _array(b, direction / np.linalg.norm(direction))
        finite = (
            objective(x + 1e-5 * direction) - objective(x - 1e-5 * direction)
        ) / 2e-5
        np.testing.assert_allclose(
            b.numpy((gradient * direction).sum()), b.numpy(finite), rtol=2e-5, atol=2e-5
        )


def test_selected_pair_rejects_malformed_indices(backend):
    b = backend.molecule(CASES["mixed_batch"], terms=("fa_ljatr",))
    pair = b.score_function.render_block_pair_scoring_module(b.pose)
    x = b.pose.coords
    with pytest.raises(ValueError, match="shape"):
        pair.score_interactions(x, _array(b, np.array([0, 1], np.int32)))
    with pytest.raises(TypeError, match="integer"):
        pair.score_interactions(x, _array(b, np.array([[0, 1]], np.float32)))
