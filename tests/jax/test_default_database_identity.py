"""Default preparation shares chemistry; explicit custom inputs remain independent."""

import shutil

import yaml

from tmol_core.chemical.default import default_patched_chemical_database
from tmol_jax import database


def test_default_parameter_context_uses_shared_chemical_identity(monkeypatch):
    directory = database.default_directory()
    monkeypatch.delenv("TMOL_JAX_DATABASE", raising=False)
    # The source checkout uses exported scoring tables; installed packages use
    # the identical packaged directory through the unmodified default accessor.
    monkeypatch.setattr(database, "default_directory", lambda: directory)

    class IsolatedDatabase(database.ParameterDatabase):
        _default = None

    context = IsolatedDatabase.get_default()
    assert context.chemical is default_patched_chemical_database()
    assert IsolatedDatabase.get_default() is context
    explicit = IsolatedDatabase.from_file(str(directory))
    assert explicit.chemical is not context.chemical
    assert [r.name for r in explicit.chemical.residues] == [
        r.name for r in context.chemical.residues
    ]


def test_explicit_default_path_keeps_custom_chemical_alias(monkeypatch, tmp_path):
    original = database.default_directory()
    shutil.copytree(original / "chemical", tmp_path / "chemical")
    (tmp_path / "scoring").symlink_to(original / "scoring", target_is_directory=True)
    chemical = tmp_path / "chemical/chemical.yaml"
    record = yaml.safe_load(chemical.read_text())
    record["name3_aliases"].append({"name3": "XAL", "read_as": "ALA"})
    chemical.write_text(yaml.safe_dump(record))
    monkeypatch.setenv("TMOL_JAX_DATABASE", str(tmp_path))

    class IsolatedDatabase(database.ParameterDatabase):
        _default = None

    context = IsolatedDatabase.get_default()
    assert context.chemical is not default_patched_chemical_database()
    assert any(
        a.name3 == "XAL" and a.read_as == "ALA" for a in context.chemical.name3_aliases
    )
    assert not any(
        a.name3 == "XAL" for a in default_patched_chemical_database().name3_aliases
    )
