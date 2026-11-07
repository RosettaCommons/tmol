"""Custom reflected library metadata survives the Torch-free table format."""

import attrs
import numpy as np

from tmol_core.database.mirrored_dunbrack import _mirror_rotameric_data
from tmol_jax.database import ParameterDatabase
from tmol_jax.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary


def test_custom_reflected_origin_survives_npz(tmp_path):
    source = ParameterDatabase.get_default().scoring.dunbrack.rotameric_libraries[0]
    original = attrs.evolve(
        source.rotameric_data,
        backbone_dihedral_start=source.rotameric_data.backbone_dihedral_start
        + np.array([1.25, -2], dtype=np.float32),
    )
    mirrored = _mirror_rotameric_data(original)
    assert mirrored.backbone_is_mirrored
    np.testing.assert_array_equal(
        mirrored.backbone_source_start, original.backbone_dihedral_start
    )
    arrays = {"__n_rot": 1, "__n_semi": 0, "rot0__table_name": "private_reflection"}
    arrays.update(
        {
            "rot0__data__" + field.name: getattr(mirrored, field.name)
            for field in attrs.fields(type(mirrored))
            if getattr(mirrored, field.name) is not None
        }
    )
    path = tmp_path / "custom.npz"
    np.savez(path, **arrays)
    (loaded,), semi = DunbrackRotamerLibrary.load_libraries_from_npz(path)
    assert not semi and loaded.table_name == "private_reflection"
    for field in attrs.fields(type(mirrored)):
        np.testing.assert_array_equal(
            getattr(loaded.rotameric_data, field.name), getattr(mirrored, field.name)
        )
    restored = _mirror_rotameric_data(loaded.rotameric_data)
    for field in attrs.fields(type(original)):
        np.testing.assert_array_equal(
            getattr(restored, field.name), getattr(original, field.name)
        )
