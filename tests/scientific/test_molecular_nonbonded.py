"""Nonbonded physics and independent per-channel derivatives across domains."""

import numpy as np
import pytest

from .molecules import DOMAIN_CASES

TERMS = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")


@pytest.mark.parametrize("name", DOMAIN_CASES)
def test_molecular_nonbonded_channels(backend, name):
    backend.molecule(DOMAIN_CASES[name], terms=TERMS)
    x = backend.coords()
    value, gradient = backend.channel_values_and_gradients(x)
    assert np.isfinite(value).all() and np.isfinite(gradient).all()
    if name == "metal_cluster":
        # SF4 has zero LJ well depth, solvation free energy and charge in this
        # database. It must remain exactly inert for these four terms.
        np.testing.assert_array_equal(value, 0)
        np.testing.assert_array_equal(gradient, 0)
        return
    assert np.all(np.abs(value) > 1e-3)
    assert np.all(np.linalg.norm(gradient.reshape(4, -1), axis=1) > 1e-3)
    direction = np.random.default_rng(194).normal(size=x.shape).astype(x.dtype)
    direction /= np.linalg.norm(direction)
    step = 5e-3
    numeric = (
        backend.scores(x + step * direction, sum_terms=False, apply_weights=False)
        - backend.scores(x - step * direction, sum_terms=False, apply_weights=False)
    ) / (2 * step)
    analytic = (gradient * direction).sum(axis=(2, 3))
    np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(gradient.sum(axis=2), 0, atol=1e-3)
