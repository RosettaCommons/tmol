"""Common Dunbrack coordinate and custom-database scientific contracts."""

import attrs
import numpy as np
import pytest

from .test_score_contract import check_coordinate_transforms

from .dunbrack_molecules import DUNBRACK_CASES
from .molecules import CASES, DOMAIN_CASES

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

    if name == "mixed_batch":
        check_coordinate_transforms(backend, DUNBRACK_CASES[name], TERMS)

    if backend.framework == "jax" and name in ("d_peptide", "mixed_batch"):
        _check_coordinate_precision(backend)


@pytest.mark.parametrize(
    "name", ["ligand", "noncanonical", "connected_noncanonical", "dna"]
)
def test_unassigned_rotamer_library_is_exactly_zero(backend, name):
    backend.molecule(DOMAIN_CASES[name], terms=TERMS)
    values, gradients = backend.channel_values_and_gradients(backend.coords())
    # HYP borrows a sampling reference; that must not invent a scoring library.
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_array_equal(gradients, 0)

    if backend.framework == "jax" and name in ("ligand", "dna"):
        _check_inactive_storage(backend, DOMAIN_CASES[name])


@pytest.mark.parametrize(
    "scenario",
    ("width", "selected", "custom_selected", "table_owners", "probabilities"),
)
def test_custom_dunbrack_width_changes_only_deviation_channel(backend, scenario):
    if scenario != "width":
        if backend.framework != "jax":
            pytest.skip("JAX selected-library preparation capability")
        if scenario in ("selected", "custom_selected"):
            _check_selected_libraries(backend, custom=scenario == "custom_selected")
            if scenario == "selected":
                _check_unselected_lookup_errors()
        elif scenario == "table_owners":
            _check_table_owners(backend)
        else:
            _check_probability_mutation(backend)
        return
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
    # Parameter contexts may share a chemical palette. The new widths must
    # replace that palette's prior scoring preparation without changing the
    # original score function's parameter snapshot.
    changed.pose = backend.pose
    if backend.framework == "torch":
        changed._render_torch()
    new_values, new_gradients = changed.channel_values_and_gradients(x)
    np.testing.assert_allclose(new_values[[0, 2]], values[[0, 2]], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        new_gradients[[0, 2]], gradients[[0, 2]], rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(new_values[1], values[1] / 4, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(new_gradients[1], gradients[1] / 4, rtol=2e-6, atol=2e-6)

    retained_values, retained_gradients = backend.channel_values_and_gradients(x)
    np.testing.assert_allclose(retained_values, values, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(retained_gradients, gradients, rtol=2e-6, atol=2e-6)


def _check_coordinate_precision(b):
    import jax
    import jax.numpy as jnp

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


def _check_inactive_storage(b, case):
    import jax
    import jax.numpy as jnp

    from tests.scientific.rotamer_execution import RotamerExecution
    from tests.scientific.test_canonical_molecular_input import (
        canonical_input,
        rebuild_canonical,
    )

    co, canonical = canonical_input(case, b.parameter_database)
    rebuild_canonical(b, co, canonical)
    palette = b.pose.packed_block_types
    assert any(rt.base_name == "LEU" for rt in palette.active_block_types)
    rotamers = RotamerExecution(b, family="dunbrack").rotamers
    whole = b.score_function.render_whole_pose_scoring_module(
        b.pose, sum_terms=False, apply_weights=False
    )
    sparse = b.score_function.render_rotamer_scoring_module(
        b.pose, rotamers, sum_terms=False, apply_weights=False
    )
    # Small inactive molecular objectives must fit a 64 KiB operand budget even
    # when the public palette retains the complete amino-acid library choices.
    for operands in (whole.arrays, sparse._arrays):
        assert sum(x.nbytes for x in jax.tree.leaves(operands)) <= 64 * 1024
    fixed = sparse.prepare(rotamers.coords)
    with jax.default_device(b.device), jax.enable_x64():
        for dtype in (jnp.float32, jnp.float64):
            for score, coords in (
                (whole, b.pose.coords.astype(dtype)),
                (
                    lambda x: fixed(x)[0],
                    rotamers.coords.astype(dtype),
                ),
            ):
                for default in (b.device, jax.devices("cpu")[0]):
                    with jax.default_device(default):
                        values = score(coords)
                        assert values.shape[0] == 3 and values.dtype == dtype
                        assert values.device == b.device
                        np.testing.assert_array_equal(values, 0)
                        # JAX prunes unused coordinates from a constant objective.
                        # Explicit output placement overrides a CPU default even
                        # when no numerical operand remains on the requested GPU.
                        placement = (
                            {"out_shardings": coords.sharding}
                            if default != b.device
                            else {}
                        )
                        evaluate = jax.jit(
                            jax.vmap(jax.value_and_grad(lambda x: score(x).sum())),
                            **placement,
                        )
                        energies, forces = evaluate(jnp.stack((coords, coords + 0.1)))
                        assert forces.shape == (2, *coords.shape)
                        assert forces.dtype == dtype and forces.device == b.device
                        np.testing.assert_array_equal(energies, 0)
                        np.testing.assert_array_equal(forces, 0)
    assert b.pose.packed_block_types is palette


def _check_table_owners(b):
    import jax

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

    from tmol_core.scoring.dunbrack import DunbrackTableData
    from tmol_jax.score.dunbrack.parameters import device_dunbrack_tables

    # A compact table snapshot isolates lifetime from the bounded host cache.
    values = np.array([0.25, 1.125], dtype=np.float32)
    tables = DunbrackTableData({"values": values}, {}, {}, {}, {}, {})
    owner = weakref.ref(tables)
    device = b.device
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


def _check_probability_mutation(b):
    import jax
    import jax.numpy as jnp

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
    b.molecule(DUNBRACK_CASES["l_peptide"], terms=TERMS, parameter_database=database)
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


def _check_selected_libraries(b, *, custom):
    import jax
    import jax.numpy as jnp
    from tmol_jax import ParameterDatabase, PackedBlockTypes
    from tmol_jax.io import pose_stack_from_canonical_form
    from tmol_jax.score.dunbrack.dunbrack_energy_term import DunbrackEnergyTerm
    from .molecules import MolecularCase
    from .test_canonical_molecular_input import canonical_input

    class _FullLibraryTerm(DunbrackEnergyTerm):
        def _packed_parameters(self, pbt, coordinate_dtype, table_indices):
            # The existing common full-library preparation and native scorer form
            # the independent storage-layout reference; no physics is reimplemented.
            return super()._packed_parameters(pbt, coordinate_dtype, None)

    original_database = database = ParameterDatabase.get_default()
    if custom:
        library = database.scoring.dunbrack
        lookup = {row.residue_name: row.dun_table_name for row in library.dun_lookup}
        replacements = dict(LEU="ILE", PHE="TYR", DLEU="DILE", DPHE="DTYR")
        library = attrs.evolve(
            library,
            dun_lookup=tuple(
                (
                    attrs.evolve(
                        row, dun_table_name=lookup[replacements[row.residue_name]]
                    )
                    if row.residue_name in replacements
                    else row
                )
                for row in library.dun_lookup
            ),
        )
        database = attrs.evolve(
            database, scoring=attrs.evolve(database.scoring, dunbrack=library)
        )
    device = b.device
    with jax.default_device(device), jax.enable_x64():
        palette = PackedBlockTypes.default(database.chemical)
        names = tuple(bt.name for bt in palette.active_block_types)
        term = DunbrackEnergyTerm(database)
        full_term = _FullLibraryTerm(database)
        retained = []
        for identity in ("LEU", "PHE", "DLEU", "DPHE", "LEU"):
            case = MolecularCase((), ((identity,),), ((1,),))
            ordering, canonical = canonical_input(case, database)
            pose = pose_stack_from_canonical_form(
                dict(canonical, canonical_ordering=ordering),
                palette,
                compact_padding=True,
            )
            pose = pose.replace(coords=pose.coords.astype(jnp.float64))
            selected, full = [t.setup_for_pose_stack(pose) for t in (term, full_term)]
            selected_bytes = sum(a.nbytes for a in selected.parameters)
            full_bytes = sum(a.nbytes for a in full.parameters)
            assert 0 < selected_bytes < full_bytes / 4

            def score(x, state):
                return jnp.stack(tuple(DunbrackEnergyTerm.evaluate(state, x).values()))

            execute = jax.jit(score)
            derivative = jax.jit(jax.jacrev(score))
            expected = execute(pose.coords, full), derivative(pose.coords, full)
            actual = execute(pose.coords, selected), derivative(pose.coords, selected)
            assert np.max(np.abs(expected[0])) > 1e-3
            assert np.linalg.norm(expected[1]) > 1e-3
            if custom and identity == "LEU":
                original = _FullLibraryTerm(original_database).setup_for_pose_stack(
                    pose
                )
                assert (
                    np.max(np.abs(execute(pose.coords, original) - expected[0])) > 1e-3
                )
            for value, reference in zip(actual, expected):
                assert value.dtype == jnp.float64 and value.device == device
                np.testing.assert_allclose(value, reference, rtol=2e-10, atol=2e-10)
            retained.append((pose.coords, selected, actual))
            assert pose.packed_block_types is palette
            assert tuple(bt.name for bt in palette.active_block_types) == names
        # Later table selections must not invalidate previously rendered states.
        for coords, state, reference in retained:
            for value, expected in zip(
                (execute(coords, state), derivative(coords, state)), reference
            ):
                np.testing.assert_array_equal(value, expected)


def _check_unselected_lookup_errors():
    from tmol_jax import ParameterDatabase

    from tmol_jax.score.dunbrack.parameters import prepare_dunbrack_parameters

    database = ParameterDatabase.get_default()
    library = database.scoring.dunbrack
    invalid = attrs.evolve(
        library.dun_lookup[0], residue_name="UNUSED", dun_table_name="missing"
    )
    for extra in (invalid, library.dun_lookup[-1]):
        source = attrs.evolve(library, dun_lookup=(*library.dun_lookup, extra))
        changed = attrs.evolve(
            database, scoring=attrs.evolve(database.scoring, dunbrack=source)
        )
        with pytest.raises(ValueError, match="missing tables|must be unique"):
            prepare_dunbrack_parameters(changed, table_indices=(0,))
