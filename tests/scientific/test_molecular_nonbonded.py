"""Nonbonded physics and independent per-channel derivatives across domains."""

import numpy as np
import pytest

from .test_score_contract import check_coordinate_transforms

from .molecules import DOMAIN_CASES

TERMS = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")


@pytest.mark.parametrize("name", DOMAIN_CASES)
def test_molecular_nonbonded_channels(backend, name):
    check_molecular_nonbonded_channels(backend, name, TERMS)


@pytest.mark.parametrize("name", DOMAIN_CASES)
def test_independent_ljlk_channels(backend, name):
    check_molecular_nonbonded_channels(backend, name, TERMS[:3])


def check_molecular_nonbonded_channels(backend, name, terms):
    backend.molecule(DOMAIN_CASES[name], terms=terms)
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
    assert np.all(np.linalg.norm(gradient.reshape(len(terms), -1), axis=1) > 1e-3)
    direction = np.random.default_rng(194).normal(size=x.shape).astype(x.dtype)
    direction /= np.linalg.norm(direction)
    # Large repulsive energies lose low bits in float32 subtraction. Keep the
    # tested analytic gradient in its original dtype, and differentiate values
    # in float64 at the exact same representable coordinates and direction.
    numeric = backend.float64_directional_derivative(x, direction)
    analytic = (gradient * direction).sum(axis=(2, 3))
    np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=3e-3)
    # Measure cancellation in float64 without changing the computed gradients.
    # Float32 summation can obscure a small net force among large pair forces.
    np.testing.assert_allclose(gradient.sum(axis=2, dtype=np.float64), 0, atol=1e-3)

    if name == "mixed_batch":
        check_coordinate_transforms(
            backend,
            DOMAIN_CASES[name],
            terms,
            empty_term="fa_ljatr" if len(terms) == 4 else None,
        )
