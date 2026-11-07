"""Weighted native totals preserve the public decomposed scoring contract."""

import numpy as np

from .molecules import CASES


def test_nonbonded_custom_weights_and_gradients(backend):
    terms = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")
    backend.molecule(CASES["mixed_batch"], terms=terms)
    x = backend.coords()
    values, gradients = backend.channel_values_and_gradients(x)
    weights = np.asarray([0.7, 0.44, 0.0, -0.3], dtype=x.dtype)
    for score_type, weight in zip(
        (
            backend.score_function.active_score_types()
            if backend.framework == "jax"
            else backend.score_function.all_score_types()
        ),
        weights,
    ):
        backend.score_function.set_weight(score_type, float(weight))
    if backend.framework == "torch":
        backend._render_torch()
    total, gradient = backend.value_and_grad(x)
    np.testing.assert_allclose(total, weights @ values, rtol=3e-6, atol=3e-5)
    np.testing.assert_allclose(
        gradient, np.einsum("s,spac->pac", weights, gradients), rtol=5e-5, atol=5e-5
    )
    contributions = backend.scores(x, sum_terms=False)
    np.testing.assert_allclose(contributions.sum(axis=0), total, rtol=3e-6, atol=3e-5)


def test_independent_nonbonded_terms_do_not_require_other_parameters(backend):
    import attrs

    case = CASES["mixed_batch"]
    terms = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")
    backend.molecule(case, terms=terms)
    x = backend.coords()
    expected, expected_gradients = backend.channel_values_and_gradients(x)
    database = backend.parameter_database
    for keep_electrostatics in (False, True):
        absent = "ljlk" if keep_electrostatics else "elec"
        field = (
            "atom_type_parameters" if keep_electrostatics else "atom_charge_parameters"
        )
        missing = attrs.evolve(getattr(database.scoring, absent), **{field: ()})
        db = attrs.evolve(
            database, scoring=attrs.evolve(database.scoring, **{absent: missing})
        )
        selected = terms[-1:] if keep_electrostatics else terms[:3]
        rows = slice(3, 4) if keep_electrostatics else slice(0, 3)
        backend.molecule(case, terms=selected, parameter_database=db)
        np.testing.assert_array_equal(backend.coords(), x)
        value, gradient = backend.channel_values_and_gradients(x)
        np.testing.assert_allclose(value, expected[rows], rtol=3e-6, atol=3e-5)
        np.testing.assert_allclose(
            gradient, expected_gradients[rows], rtol=5e-5, atol=5e-5
        )
