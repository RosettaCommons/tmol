"""Noncanonical water-mediated solvation and composed coordinate derivatives."""

import attrs
import numpy as np
import pytest

from .molecules import DOMAIN_CASES

TERMS = ("lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl")
CASES = ("ligand", "connected_noncanonical", "mixed_batch", "d_peptide", "modified_dna")


@pytest.mark.parametrize("name", CASES)
def test_lk_ball_molecular_channels(backend, name):
    backend.molecule(DOMAIN_CASES[name], terms=TERMS)
    coords = backend.coords()
    values, gradients = backend.channel_values_and_gradients(coords)
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    assert np.all(np.abs(values[:2]) > 1e-4)
    if name in ("connected_noncanonical", "modified_dna"):
        assert np.all(np.abs(values[2:]) > 1e-4)
        assert np.all(np.linalg.norm(gradients.reshape(4, -1), axis=1) > 1e-4)
    direction = (
        np.random.default_rng(411).normal(size=coords.shape).astype(coords.dtype)
    )
    direction /= np.linalg.norm(direction)
    step = 1e-3
    plus = backend.scores(
        coords + step * direction, sum_terms=False, apply_weights=False
    )
    minus = backend.scores(
        coords - step * direction, sum_terms=False, apply_weights=False
    )
    np.testing.assert_allclose(
        (plus - minus) / (2 * step),
        (gradients * direction).sum(axis=(2, 3)),
        rtol=5e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(gradients.sum(axis=2), 0, atol=2e-5)


def test_lk_ball_free_energy_scaling_leaves_uncoupled_bridge_unchanged(backend):
    case = DOMAIN_CASES["connected_noncanonical"]
    backend.molecule(case, terms=TERMS)
    coords = backend.coords()
    values, gradients = backend.channel_values_and_gradients(coords)
    database = backend.parameter_database
    lk = attrs.evolve(
        database.scoring.ljlk,
        atom_type_parameters=tuple(
            attrs.evolve(row, lk_dgfree=2 * row.lk_dgfree)
            for row in database.scoring.ljlk.atom_type_parameters
        ),
    )
    scaled = attrs.evolve(database, scoring=attrs.evolve(database.scoring, ljlk=lk))
    backend.molecule(case, terms=TERMS, parameter_database=scaled)
    new_values, new_gradients = backend.channel_values_and_gradients(coords)
    np.testing.assert_allclose(new_values[:3], 2 * values[:3], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        new_gradients[:3], 2 * gradients[:3], rtol=2e-6, atol=2e-6
    )
    np.testing.assert_array_equal(new_values[3], values[3])
    np.testing.assert_array_equal(new_gradients[3], gradients[3])


def test_nonpolar_metal_cluster_has_no_water_terms(backend):
    backend.molecule(DOMAIN_CASES["metal_cluster"], terms=TERMS)
    values, gradients = backend.channel_values_and_gradients(backend.coords())
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_array_equal(gradients, 0)
