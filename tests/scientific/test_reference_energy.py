"""Reference energies preserve chemical identity and explicit D overrides."""

import attrs
import numpy as np
import pytest

from .molecules import CASES, MolecularCase


@pytest.mark.parametrize("explicit_d", [False, True])
def test_reference_identity_and_parameter_context(backend, explicit_d):
    case = MolecularCase(
        CASES["mixed_batch"].parameter_files,
        (("DALA:nterm", "DLEU:cterm", "HYP", "LG1"), ("ALA", "LEU")),
        ((2, 1, 1), (2,)),
    )
    backend.molecule(case, terms=("ref",))
    original_database = backend.parameter_database
    # Deliberately omit DALA: it must inherit ALA. An explicit DLEU value wins.
    weights = {"ALA": 2.0, "LEU": -3.0, "HYP": 1.25, "LG1": -0.75}
    if explicit_d:
        weights["DLEU"] = 5.0
    database = attrs.evolve(
        original_database,
        scoring=attrs.evolve(
            original_database.scoring,
            ref=attrs.evolve(original_database.scoring.ref, weights=weights),
        ),
    )
    backend.molecule(case, terms=("ref",), parameter_database=database)
    xyz = backend.coords()
    expected = np.array([7.5 if explicit_d else -0.5, -1.0])
    value, gradient = backend.value_and_grad(xyz, coordinate_independent=True)
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(gradient, np.zeros_like(xyz))
    np.testing.assert_array_equal(backend.scores(xyz + 7.0), expected)
    # Reusing a palette with a different parameter context must not retain it.
    changed = attrs.evolve(
        database,
        scoring=attrs.evolve(
            database.scoring,
            ref=attrs.evolve(database.scoring.ref, weights={"ALA": 4.0}),
        ),
    )
    backend.molecule(case, terms=("ref",), parameter_database=changed)
    changed_value, changed_gradient = backend.value_and_grad(
        xyz, coordinate_independent=True
    )
    np.testing.assert_array_equal(changed_value, [4.0, 4.0])
    np.testing.assert_array_equal(changed_gradient, np.zeros_like(xyz))
    backend.set_options({"ref_weights": {"ALA": 6.0, "DLEU": 0.0}})
    np.testing.assert_array_equal(backend.scores(xyz), [6.0, 6.0])
    backend.set_options({})
    np.testing.assert_array_equal(backend.scores(xyz), [4.0, 4.0])
