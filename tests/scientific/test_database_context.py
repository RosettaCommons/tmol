"""Framework storage adapters preserve shared custom-database semantics."""

import importlib
import os

import attrs
import numpy as np
import pytest


@pytest.fixture(
    scope="module",
    params=sorted(
        {
            lane.split("-")[0]
            for lane in os.environ.get("TMOL_TEST_BACKENDS", "jax-cpu,torch-cpu").split(
                ","
            )
        }
    ),
)
def database(request):
    pytest.importorskip(request.param)
    package = "tmol" if request.param == "torch" else "tmol_jax"
    return importlib.import_module(
        package + ".database"
    ).ParameterDatabase.get_default()


def test_context_is_immutable_and_subtypes_have_private_defaults(database):
    class CustomContext(type(database)):
        pass

    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        database.chemical = None
    custom = CustomContext.get_default()
    assert type(custom) is CustomContext
    assert custom is CustomContext.get_default() and custom is not database
    subset = custom.create_stable_subset(["MET", "GLY"], [])
    assert type(subset) is CustomContext and subset.scoring is custom.scoring
    assert [r.name for r in subset.chemical.residues] == ["MET", "GLY"]
    assert any(
        a.name3 == "MSE" and a.read_as == "MET" for a in subset.chemical.name3_aliases
    )
    with pytest.raises(ValueError, match="UNKNOWN_RESIDUE"):
        custom.create_stable_subset(["UNKNOWN_RESIDUE"], [])


def test_symmetric_gly_updates_coupled_geometry_and_parameter_identity(database):
    modified = database.with_symmetric_gly()
    repeated = modified.with_symmetric_gly()
    for name in ("rama", "omega_bbdep"):
        source, changed, twice = (
            getattr(db.scoring, name) for db in (database, modified, repeated)
        )
        assert source.uniq_id == source.content_id()
        assert (
            source.uniq_id != changed.uniq_id == twice.uniq_id == changed.content_id()
        )
    source = next(r for r in database.chemical.residues if r.name == "GLY")
    changed = next(r for r in modified.chemical.residues if r.name == "GLY")
    old = {i.name: i.d for i in source.icoors}
    new = {i.name: i.d for i in changed.icoors}
    assert old["HA2"] != old["HA3"]
    assert new["HA2"] == new["HA3"] == (old["HA2"] + old["HA3"]) / 2
    assert database.scoring.cartbonded.hash != modified.scoring.cartbonded.hash
    assert modified.scoring.cartbonded.hash == repeated.scoring.cartbonded.hash


def test_numpy_and_tensor_parameter_content_identity(database):
    # Cross-storage contents have one identity; scalar metadata and every byte
    # contribute. A mutation in the middle of a large array must invalidate it.
    from tmol_core.database.content_hash import _content_hash

    source = database.scoring.rama
    table = source.rama_tables[0]
    array = np.asarray(table.table)
    records = tuple(
        attrs.evolve(t, table=np.asarray(t.table)) for t in source.rama_tables
    )
    assert _content_hash(source.rama_lookup, records) == source.uniq_id
    changed = array.copy()
    changed.flat[changed.size // 2] += np.float32(0.25)
    records = (attrs.evolve(table, table=changed), *records[1:])
    assert _content_hash(source.rama_lookup, records) != source.uniq_id


def test_d_residue_lookup_uses_actual_mirrored_library(database):
    library = getattr(database.scoring, database._dunbrack_attribute)
    lookup = {r.residue_name: r.dun_table_name for r in library.dun_lookup}
    tables = {t.table_name: t.rotameric_data for t in library.rotameric_libraries}
    assert {"VAL", "DVAL"} <= lookup.keys()
    original, mirrored = (tables[lookup[name]] for name in ("VAL", "DVAL"))
    assert mirrored.backbone_is_mirrored and not original.backbone_is_mirrored
    left, right = np.asarray(original.rotamer_probabilities), np.asarray(
        mirrored.rotamer_probabilities
    )
    # Default periodic -180/10 degree grid: physical reflection maps k to -k.
    rows, columns = (-np.arange(left.shape[1])) % left.shape[1], (
        -np.arange(left.shape[2])
    ) % left.shape[2]
    np.testing.assert_array_equal(right, left[:, rows][:, :, columns])
    means = np.asarray(original.rotamer_means)
    np.testing.assert_array_equal(
        np.asarray(mirrored.rotamer_means), -means[:, rows][:, :, columns]
    )


def test_bundled_backbone_parameter_identity_across_frameworks():
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    from tmol.database import ParameterDatabase as TorchDatabase
    from tmol_jax.database import ParameterDatabase as JaxDatabase

    torch_db, jax_db = TorchDatabase.get_default(), JaxDatabase.get_default()
    for name in ("rama", "omega_bbdep"):
        left, right = getattr(torch_db.scoring, name), getattr(jax_db.scoring, name)
        # This includes lookup rows, grid metadata and every coefficient-source
        # byte, so generated variant mappings cannot disappear during export.
        assert left.content_id() == right.content_id()
