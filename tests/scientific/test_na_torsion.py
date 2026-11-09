"""Both NA channels, their coordinate derivatives and independent parameters."""

import attrs
import numpy as np
import pytest

from .test_score_contract import check_coordinate_transforms

from .execution import Execution
from .na_molecules import NA_CASES

TERMS = ("na_torsion", "na_torsion_well")


@pytest.mark.parametrize("name", NA_CASES)
def test_na_channels_and_coordinate_derivatives(backend, name):
    backend.molecule(NA_CASES[name], terms=TERMS)
    x = backend.coords()
    values, jacobian = backend.channel_values_and_gradients(x)
    assert np.isfinite(values).all() and np.isfinite(jacobian).all()
    assert np.all(values > 1e-3)
    assert np.all(np.linalg.norm(jacobian.reshape(2, -1), axis=1) > 1e-3)
    direction = np.random.default_rng(908).normal(size=x.shape).astype(x.dtype)
    direction /= np.linalg.norm(direction)
    step = 5e-3
    numeric = (
        backend.scores(x + step * direction, sum_terms=False, apply_weights=False)
        - backend.scores(x - step * direction, sum_terms=False, apply_weights=False)
    ) / (2 * step)
    analytic = (jacobian * direction).sum(axis=(-2, -1))
    np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(jacobian.sum(axis=-2), 0, atol=2e-3)

    if name == "mixed_batch":
        check_coordinate_transforms(
            backend, NA_CASES[name], TERMS, empty_term="na_torsion"
        )


def test_na_custom_widths_affect_deviation_not_population_channel(backend):
    case = NA_CASES["methylcytosine"]
    backend.molecule(case, terms=TERMS)
    value, gradient = backend.channel_values_and_gradients(backend.coords())
    db = backend.parameter_database
    params = db.scoring.na_torsion
    scaled = attrs.evolve(
        params,
        global_parameters={
            name: attrs.evolve(
                row,
                sdev_sugar=2 * row.sdev_sugar,
                sdev_chi=2 * row.sdev_chi,
                sdev_backbone=tuple(2 * x for x in row.sdev_backbone),
            )
            for name, row in params.global_parameters.items()
        },
    )
    changed = attrs.evolve(db, scoring=attrs.evolve(db.scoring, na_torsion=scaled))
    other = Execution(backend.name).molecule(
        case, terms=TERMS, parameter_database=changed
    )
    new_value, new_gradient = other.channel_values_and_gradients(other.coords())
    np.testing.assert_allclose(new_value[0], value[0] / 4, rtol=3e-6, atol=3e-5)
    np.testing.assert_allclose(new_gradient[0], gradient[0] / 4, rtol=3e-6, atol=3e-5)
    np.testing.assert_array_equal(new_value[1], value[1])
    np.testing.assert_array_equal(new_gradient[1], gradient[1])
    np.testing.assert_array_equal(
        backend.scores(backend.coords(), sum_terms=False, apply_weights=False), value
    )


def test_na_explicit_unknown_base_reference_preserves_modified_chemistry(backend):
    case = NA_CASES["methylcytosine"]
    backend.molecule(case, terms=TERMS)
    db = backend.parameter_database
    changed = attrs.evolve(
        db,
        chemical=attrs.evolve(
            db.chemical,
            residues=tuple(
                attrs.evolve(row, na_base_reference="DX") if row.name == "5CM" else row
                for row in db.chemical.residues
            ),
        ),
    )
    other = Execution(backend.name).molecule(
        case, terms=TERMS, parameter_database=changed
    )
    np.testing.assert_array_equal(backend.coords(), other.coords())
    modified = next(
        bt
        for bt in other.pose.packed_block_types.active_block_types
        if bt.name == "5CM"
    )
    assert modified.na_base_reference == "DX"
    assert "C5A" in modified.atom_to_idx
    old_values, old_gradient = backend.channel_values_and_gradients(backend.coords())
    new_values, new_gradient = other.channel_values_and_gradients(other.coords())
    assert np.isfinite(new_values).all() and np.isfinite(new_gradient).all()
    assert np.linalg.norm(old_values - new_values) > 1e-3
    assert np.linalg.norm(old_gradient - new_gradient) > 1e-3
