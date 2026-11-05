"""Weighted native totals preserve the public decomposed scoring contract."""

import numpy as np

from .molecules import CASES


def test_nonbonded_custom_weights_and_gradients(backend):
    terms = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")
    backend.molecule(CASES["mixed_batch"], terms=terms)
    x = backend.coords()
    values, gradients = backend.channel_values_and_gradients(x)
    weights = np.asarray([0.7, 0.44, 0.0, -0.3], dtype=x.dtype)
    for score_type, weight in zip(
        (
            backend.score_function.active_score_types()
            if backend.framework == "jax"
            else backend.score_function.all_score_types()
        ),
        weights,
    ):
        backend.score_function.set_weight(score_type, float(weight))
    if backend.framework == "torch":
        backend._render_torch()
    total, gradient = backend.value_and_grad(x)
    np.testing.assert_allclose(total, weights @ values, rtol=3e-6, atol=3e-5)
    np.testing.assert_allclose(
        gradient, np.einsum("s,spac->pac", weights, gradients), rtol=5e-5, atol=5e-5
    )
    contributions = backend.scores(x, sum_terms=False)
    np.testing.assert_allclose(contributions.sum(axis=0), total, rtol=3e-6, atol=3e-5)
