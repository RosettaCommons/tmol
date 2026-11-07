"""Actual hydrogen bonds in ligand, D/beta-amino-acid and modified DNA systems."""

import attrs
import numpy as np
import pytest

from .hbond_molecules import HBOND_CASES


@pytest.mark.parametrize("name", HBOND_CASES)
def test_noncanonical_hbond_gradient_and_custom_donor_weights(backend, name):
    case = HBOND_CASES[name]
    backend.molecule(case, terms=("hbond",))
    coordinates = backend.coords()
    values, gradients = backend.value_and_grad(coordinates)
    assert np.all(values < -0.1)
    assert np.linalg.norm(gradients) > 0.1
    direction = (
        np.random.default_rng(29).normal(size=coordinates.shape).astype(np.float32)
    )
    direction /= np.linalg.norm(direction)
    numeric = backend.float64_directional_derivative(coordinates, direction)
    analytic = (gradients * direction).sum(axis=(1, 2))
    np.testing.assert_allclose(analytic, numeric.reshape(-1), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(gradients.sum(axis=1), 0, atol=2e-5)
    database = backend.parameter_database
    hbond = database.scoring.hbond
    changed = attrs.evolve(
        database,
        scoring=attrs.evolve(
            database.scoring,
            hbond=attrs.evolve(
                hbond,
                donor_type_params=tuple(
                    attrs.evolve(donor, weight=2 * donor.weight)
                    for donor in hbond.donor_type_params
                ),
            ),
        ),
    )
    backend.use_parameter_database(changed)
    scaled_values, scaled_gradients = backend.value_and_grad(coordinates)
    np.testing.assert_allclose(scaled_values, 2 * values, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(scaled_gradients, 2 * gradients, rtol=2e-5, atol=2e-5)
    backend.use_parameter_database(database)
    restored = backend.value_and_grad(coordinates)
    for actual, expected in zip(restored, (values, gradients)):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_hbond_custom_family_index_above_byte_range(backend):
    """Equivalent donor families keep their identity beyond 255 table rows."""
    case = HBOND_CASES["ligand_pair"]
    backend.molecule(case, terms=("hbond",))
    coords = backend.coords()
    values, gradients = backend.value_and_grad(coords)
    database = backend.parameter_database
    hb = database.scoring.hbond
    block = backend.pose.packed_block_types.active_block_types[0]
    atom_type = block.atoms[block.atom_to_idx[case.donor[0]]].atom_type
    family = next(row.donor_type for row in hb.donor_atom_types if row.d == atom_type)
    original = next(row for row in hb.donor_type_params if row.name == family)
    names = tuple(f"custom_donor_{i}" for i in range(257))
    pairs = tuple(row for row in hb.pair_parameters if row.donor_type == family)
    changed_hb = attrs.evolve(
        hb,
        donor_type_params=hb.donor_type_params
        + tuple(attrs.evolve(original, name=name) for name in names),
        pair_parameters=hb.pair_parameters
        + tuple(attrs.evolve(row, donor_type=name) for name in names for row in pairs),
        donor_atom_types=tuple(
            attrs.evolve(row, donor_type=names[-1]) if row.d == atom_type else row
            for row in hb.donor_atom_types
        ),
    )
    changed = attrs.evolve(
        database, scoring=attrs.evolve(database.scoring, hbond=changed_hb)
    )
    backend.molecule(case, terms=("hbond",), parameter_database=changed)
    actual, actual_gradient = backend.value_and_grad(coords)
    np.testing.assert_array_equal(actual, values)
    np.testing.assert_array_equal(actual_gradient, gradients)
