"""Nonzero generic bonded physics on the same molecules in every runtime."""

import numpy as np
import pytest

from .test_score_contract import check_coordinate_transforms

from .molecules import BACKBONE_CASES, CASES


@pytest.mark.parametrize("name", CASES)
def test_generic_bonded_molecular_derivative(backend, name):
    backend.molecule(CASES[name])
    coords = backend.coords()
    value, gradient = backend.value_and_grad(coords)
    assert np.isfinite(value).all() and np.isfinite(gradient).all()
    assert np.all(np.abs(value) > 1e-3)
    assert np.linalg.norm(gradient) > 1e-3
    # Central differences are independent of the common analytic kernel.
    direction = np.random.default_rng(91).normal(size=coords.shape).astype(coords.dtype)
    direction /= np.linalg.norm(direction)
    # At float32, 1e-3 steps lose digits in summed ligand torsions. A
    # 5e-3 step converges with the double-precision directional derivative.
    step = 5e-3
    numeric = (
        backend.scores(coords + step * direction)
        - backend.scores(coords - step * direction)
    ) / (2 * step)
    analytic = (gradient * direction).sum(axis=(1, 2))
    np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=3e-3)
    # Each molecular component obeys translation invariance independently.
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=2e-4)
    np.testing.assert_allclose(
        backend.scores(coords + np.array([1, -2, 3], coords.dtype)),
        value,
        rtol=2e-5,
        atol=2e-4,
    )

    if name == "mixed_batch":
        check_coordinate_transforms(
            backend,
            CASES[name],
            ("gen_torsions",),
            empty_term="gen_torsions",
            empty_case=BACKBONE_CASES["mixed"],
        )
