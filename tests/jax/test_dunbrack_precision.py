"""Rendered rotamer objectives retain precision across mixed coordinate calls."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.dunbrack_molecules import DUNBRACK_CASES
from tests.scientific.execution import Execution
from tests.scientific.test_dunbrack import TERMS


@pytest.mark.parametrize("name", ("d_peptide", "mixed_batch"))
def test_rotamer_preparation_preserves_cross_dtype_values_and_gradients(name):
    backend = (
        "jax-cuda"
        if "jax-cuda" in os.environ.get("TMOL_TEST_BACKENDS", "")
        else "jax-cpu"
    )
    b = Execution(backend).molecule(DUNBRACK_CASES[name], terms=TERMS)
    with jax.default_device(b.device), jax.enable_x64():
        x32 = b.pose.coords.astype(jnp.float32)
        x64 = x32.astype(jnp.float64)
        original = b.score_function.jitted_value_and_grad(b.pose.replace(coords=x32))
        expected = original(x64)
        assert expected[0].dtype == expected[1].dtype == jnp.float64
        assert np.linalg.norm(expected[1]) > 1e-3
        prepared = b.score_function.jitted_value_and_grad(b.pose.replace(coords=x64))
        for actual, reference in zip(prepared(x64), expected):
            np.testing.assert_allclose(actual, reference, rtol=2e-10, atol=2e-10)
        batched = jax.jit(jax.vmap(prepared))(jnp.stack([x64, x64]))
        for actual, reference in zip(batched, expected):
            np.testing.assert_allclose(
                actual, jnp.stack([reference, reference]), rtol=2e-10, atol=2e-10
            )
        # A rendered scorer can still execute either coordinate dtype. Returning
        # to single precision must retain the source tables and old snapshots.
        for actual, reference in zip(prepared(x32), original(x32)):
            assert actual.dtype == reference.dtype == jnp.float32
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        restored = b.score_function.jitted_value_and_grad(b.pose.replace(coords=x32))
        for actual, reference in zip(restored(x64), expected):
            np.testing.assert_allclose(actual, reference, rtol=2e-10, atol=2e-10)


def test_fitted_tables_do_not_retain_source_library_and_sampler_owns_snapshot():
    import attrs
    import gc
    import weakref

    from tmol_jax import ParameterDatabase
    from tmol_jax.pack import DunbrackChiSampler
    from tmol_jax.score.dunbrack.parameters import prepare_dunbrack_parameters

    original = ParameterDatabase.get_default()
    library = attrs.evolve(original.scoring.dunbrack)
    database = attrs.evolve(
        original, scoring=attrs.evolve(original.scoring, dunbrack=library)
    )
    owner = weakref.ref(library)
    tables = prepare_dunbrack_parameters(database)
    sampler = DunbrackChiSampler.from_database(database)
    # Samplers own their mutable table arrays even when preparation is shared.
    for key, values in tables.scoring_db.items():
        if isinstance(values, np.ndarray):
            np.testing.assert_array_equal(sampler.tables.scoring_db[key], values)
            assert not np.shares_memory(sampler.tables.scoring_db[key], values)
    del sampler, database, library
    gc.collect()
    assert owner() is None
    assert any(np.asarray(value).size for value in tables.scoring_db.values())
