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


def test_fitted_library_cache_observes_mutable_custom_probabilities():
    import attrs

    from tmol_jax import ParameterDatabase, ScoreFunction, ScoreType

    database = DUNBRACK_CASES["l_peptide"].database(ParameterDatabase.get_default())
    source = database.scoring.dunbrack
    name = next(
        row.dun_table_name for row in source.dun_lookup if row.residue_name == "LEU"
    )
    libraries = []
    for library in source.rotameric_libraries:
        if library.table_name == name:
            probabilities = library.rotameric_data.rotamer_probabilities.copy()
            library = attrs.evolve(
                library,
                rotameric_data=attrs.evolve(
                    library.rotameric_data, rotamer_probabilities=probabilities
                ),
            )
        libraries.append(library)
    library = attrs.evolve(source, rotameric_libraries=tuple(libraries))
    database = attrs.evolve(
        database, scoring=attrs.evolve(database.scoring, dunbrack=library)
    )
    device = "cuda" if jax.devices()[0].platform == "gpu" else "cpu"
    b = Execution(f"jax-{device}").molecule(
        DUNBRACK_CASES["l_peptide"], terms=TERMS, parameter_database=database
    )
    with jax.default_device(b.device), jax.enable_x64():
        coords = b.pose.coords.astype(jnp.float64)
        original = b.score_function.jitted_value_and_grad(b.pose)
        before = original(coords)
        assert np.linalg.norm(before[1]) > 1e-3
        probabilities *= 0.5
        changed = ScoreFunction(database)
        for term in TERMS:
            changed.set_weight(getattr(ScoreType, term), 1.0)
        after = changed.jitted_value_and_grad(b.pose)(coords)
        # One LEU contributes -log(p); scaling all its probabilities adds log(2)
        # without changing the physical coordinate derivative.
        np.testing.assert_allclose(after[0] - before[0], np.log(2), atol=2e-5)
        np.testing.assert_allclose(after[1], before[1], rtol=2e-5, atol=2e-5)
        for actual, expected in zip(original(coords), before):
            np.testing.assert_array_equal(actual, expected)


def test_device_tables_do_not_retain_fitted_host_snapshot():
    import gc
    import weakref

    from tmol_core.scoring.dunbrack import DunbrackTableData
    from tmol_jax.score.dunbrack.parameters import device_dunbrack_tables

    # A compact table snapshot isolates lifetime from the bounded host cache.
    values = np.array([0.25, 1.125], dtype=np.float32)
    tables = DunbrackTableData({"values": values}, {}, {}, {}, {}, {})
    owner = weakref.ref(tables)
    device = jax.devices()[0]
    with jax.enable_x64():
        prepared32 = device_dunbrack_tables(tables, np.float32, device)
        prepared64 = device_dunbrack_tables(tables, np.float64, device)
        del tables
        gc.collect()
        assert owner() is None
        for prepared, dtype in [(prepared32, np.float32), (prepared64, np.float64)]:
            assert prepared[0].dtype == dtype
            assert prepared[0].device == device
            np.testing.assert_array_equal(prepared[0], values)
