"""Shared Cartesian-bonded physics on real molecular topologies."""

import numpy as np
import pytest

from .molecules import DOMAIN_CASES, MolecularCase

CARTESIAN_CASES = {
    **DOMAIN_CASES,
    "hydroxyls": MolecularCase((), (("SER", "THR", "TYR"),), ((3,),)),
}

TERMS = (
    "cart_lengths",
    "cart_angles",
    "cart_torsions",
    "cart_impropers",
    "cart_hxltorsions",
)


@pytest.mark.parametrize("name", CARTESIAN_CASES)
def test_cartesian_bonded_channels_and_coordinate_derivatives(backend, name):
    backend.molecule(CARTESIAN_CASES[name], terms=TERMS)
    x = backend.coords()
    values, gradients = backend.channel_values_and_gradients(x)
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    assert np.abs(values).sum() > 1e-3
    assert np.linalg.norm(gradients) > 1e-3
    # All five channels are checked, including genuinely absent interactions.
    direction = np.random.default_rng(195).normal(size=x.shape).astype(x.dtype)
    direction /= np.linalg.norm(direction)
    step = 5e-3
    numeric = (
        backend.scores(x + step * direction, sum_terms=False, apply_weights=False)
        - backend.scores(x - step * direction, sum_terms=False, apply_weights=False)
    ) / (2 * step)
    analytic = (gradients * direction).sum(axis=(2, 3))
    np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(gradients.sum(axis=2), 0, atol=1e-3)
