"""Custom statistical lookups survive both generations of Torch archive paths."""

import sys
import types

import pytest
import torch

from tmol.database.scoring._content_hash import _content_hash
from tmol.database.scoring._dunbrack_libraries import DunbrackRotamerLibrary
from tmol.database.scoring._omega_bbdep import OmegaBBDepDatabase
from tmol.database.scoring._rama import RamaDatabase
from tmol_core.database.library_records import (
    DunMappingParams,
    OmegaBBDepMappingParams,
    RamaMappingParams,
)


@pytest.mark.parametrize("namespace", ["common", "private", "historical"])
@pytest.mark.parametrize(
    "record, database, module",
    [
        (RamaMappingParams, RamaDatabase, "rama"),
        (OmegaBBDepMappingParams, OmegaBBDepDatabase, "omega_bbdep"),
        (DunMappingParams, DunbrackRotamerLibrary, "dunbrack_libraries"),
    ],
)
def test_custom_lookup_archive_paths(
    namespace, record, database, module, tmp_path, monkeypatch
):
    if record is DunMappingParams:
        original = database((record("custom_rotamers", "DPRIVATE"),), (), ())
    else:
        original = database(
            "", (record("custom_grid", "PRIVATE", "DPARTNER", True, False),), ()
        )
    expected = _content_hash(original)
    path = tmp_path / "custom.pt"
    with monkeypatch.context() as patch:
        if namespace != "common":
            prefix = "_" if namespace == "private" else ""
            legacy_path = "tmol.database.scoring." + prefix + module
            if legacy_path not in sys.modules:
                legacy = types.ModuleType(legacy_path)
                setattr(legacy, record.__name__, record)
                patch.setitem(sys.modules, legacy_path, legacy)
            patch.setattr(record, "__module__", legacy_path)
        torch.save(original, path)
    loaded = database.from_file(path)
    # Rama/omega loaders deliberately replace a stale id with the content id.
    if hasattr(loaded, "uniq_id"):
        import attr

        loaded = attr.evolve(loaded, uniq_id="")
    assert _content_hash(loaded) == expected
