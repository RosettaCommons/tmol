"""Common Dunbrack coordinate and custom-database scientific contracts."""

import attrs
import numpy as np
import pytest

from .dunbrack_molecules import DUNBRACK_CASES
from .molecules import CASES

TERMS = ("dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot")


@pytest.mark.parametrize("name", DUNBRACK_CASES)
def test_dunbrack_channels_and_coordinate_derivatives(backend, name):
    backend.molecule(DUNBRACK_CASES[name], terms=TERMS)
    x = backend.coords()
    values, gradients = backend.channel_values_and_gradients(x)
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    assert np.all(np.abs(values).sum(axis=1) > 1e-3)
    assert np.all(np.linalg.norm(gradients.reshape(3, -1), axis=1) > 1e-3)
    direction = np.random.default_rng(826).normal(size=x.shape).astype(x.dtype)
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


@pytest.mark.parametrize("name", ["ligand", "noncanonical", "connected_noncanonical"])
def test_unassigned_rotamer_library_is_exactly_zero(backend, name):
    backend.molecule(CASES[name], terms=TERMS)
    values, gradients = backend.channel_values_and_gradients(backend.coords())
    # HYP borrows a sampling reference; that must not invent a scoring library.
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_array_equal(gradients, 0)


def test_custom_dunbrack_width_changes_only_deviation_channel(backend):
    case = DUNBRACK_CASES["d_peptide"]
    backend.molecule(case, terms=TERMS)
    x = backend.coords()
    values, gradients = backend.channel_values_and_gradients(x)
    database = backend.parameter_database
    field = (
        "dunbrack" if "dunbrack" in attrs.fields_dict(type(database.scoring)) else "dun"
    )
    library = getattr(database.scoring, field)

    def widen(record):
        data = record.rotameric_data
        return attrs.evolve(
            record,
            rotameric_data=attrs.evolve(data, rotamer_stdvs=data.rotamer_stdvs * 2),
        )

    custom = attrs.evolve(
        database,
        scoring=attrs.evolve(
            database.scoring,
            **{
                field: attrs.evolve(
                    library,
                    rotameric_libraries=tuple(
                        widen(r) for r in library.rotameric_libraries
                    ),
                    semi_rotameric_libraries=tuple(
                        widen(r) for r in library.semi_rotameric_libraries
                    ),
                )
            },
        ),
    )
    changed = type(backend)(backend.name).molecule(
        case, terms=TERMS, parameter_database=custom
    )
    new_values, new_gradients = changed.channel_values_and_gradients(x)
    np.testing.assert_allclose(new_values[[0, 2]], values[[0, 2]], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        new_gradients[[0, 2]], gradients[[0, 2]], rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(new_values[1], values[1] / 4, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(new_gradients[1], gradients[1] / 4, rtol=2e-6, atol=2e-6)
