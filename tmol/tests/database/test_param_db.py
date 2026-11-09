from pathlib import Path

import tmol.database

from tmol import ParameterDatabase
from tmol.database.scoring import ScoringDatabase


def test_get_default():
    param_db1 = ParameterDatabase.get_default()
    param_db2 = ParameterDatabase.get_default()
    assert param_db1 is param_db2


def test_scoring_database_ignores_forced_unsafe_torch_load(monkeypatch):
    """RF-style process settings must not disable legacy pickle aliases."""
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    scoring_path = Path(tmol.database.__file__).parent / "default" / "scoring"
    scoring = ScoringDatabase.from_file(str(scoring_path))
    assert scoring.dun.rotameric_libraries
    assert scoring.rama.rama_tables
    assert scoring.omega_bbdep.bbdep_omega_lookup


def test_create_stable_subset(default_database):
    rt_name_subset = ["ALA", "ASN", "ARG", "PHE"]
    variant_subset = ["nterm", "cterm"]
    subset = default_database.create_stable_subset(rt_name_subset, variant_subset)
    assert len(subset.chemical.residues) == 16


def test_create_stable_subset_error_handline(default_database):
    rt_name_subset = ["ALA", "ASN", "ARG", "FEE"]
    variant_subset = ["nterm", "cterm"]
    threw = False
    try:
        default_database.create_stable_subset(rt_name_subset, variant_subset)
    except ValueError as e:
        threw = True
        gold_error = "ERROR: could not build the requested PachedChemcialDatabase subset because 'FEE' is not present in the original set"
        assert str(e) == gold_error
    assert threw


def test_parameter_context_preserves_subtype_and_default_isolation():
    class CustomContext(ParameterDatabase):
        pass

    original = ParameterDatabase.get_default()
    custom = CustomContext.get_default()
    assert type(custom) is CustomContext
    assert custom is CustomContext.get_default()
    assert custom is not original
    assert ParameterDatabase.get_default() is original
    subset = custom.create_stable_subset(["MET", "GLY"], [])
    assert type(subset) is CustomContext
    assert subset.scoring is custom.scoring
    assert all(
        alias.read_as in {"MET", "GLY"} for alias in subset.chemical.name3_aliases
    )
    assert any(alias.name3 == "MSE" for alias in subset.chemical.name3_aliases)
    symmetric = subset.with_symmetric_gly()
    assert type(symmetric) is CustomContext
    assert symmetric is not subset
