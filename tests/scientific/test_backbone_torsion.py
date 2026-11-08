"""Connected backbone values and derivatives across shared molecular cases."""

import numpy as np
import pytest

from .test_score_contract import check_coordinate_transforms

from .molecules import BACKBONE_CASES, DOMAIN_CASES


@pytest.mark.parametrize("name", BACKBONE_CASES)
def test_connected_backbone_channels_and_coordinate_derivatives(backend, name):
    backend.molecule(BACKBONE_CASES[name], terms=("rama", "omega"))
    x = backend.coords()
    values, gradients = backend.channel_values_and_gradients(x)
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    assert np.all(np.abs(values).sum(axis=1) > 1e-3)
    assert np.all(np.linalg.norm(gradients.reshape(2, -1), axis=1) > 1e-3)
    direction = np.random.default_rng(827).normal(size=x.shape).astype(x.dtype)
    direction /= np.linalg.norm(direction)
    step = 5e-3
    numeric = (
        backend.scores(x + step * direction, sum_terms=False, apply_weights=False)
        - backend.scores(x - step * direction, sum_terms=False, apply_weights=False)
    ) / (2 * step)
    np.testing.assert_allclose(
        (gradients * direction).sum(axis=(2, 3)), numeric, rtol=3e-3, atol=3e-3
    )
    np.testing.assert_allclose(gradients.sum(axis=2), 0, atol=2e-4)

    if name == "mixed":
        check_coordinate_transforms(
            backend, BACKBONE_CASES[name], ("rama", "omega"), empty_term="rama"
        )


def test_absent_backbone_reference_does_not_invent_peptide_energy(backend):
    backend.molecule(DOMAIN_CASES["connected_noncanonical"], terms=("rama", "omega"))
    values, gradients = backend.channel_values_and_gradients(backend.coords())
    # B3Q/B3D have real bonded parameters and peptide connections, but their
    # committed records assign no backbone statistical reference.
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_array_equal(gradients, 0)
