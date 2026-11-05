"""Packing chemistry is independent of residue storage order and frameworks."""

import numpy as np
import pytest

from tmol_core.chemical import ChemicalDatabase
from tmol_core.chemical.patching import PatchedChemicalDatabase
from tmol_core.chemical.restypes import refined_from_raw
from tmol_core.pack.palette import prepare_packer_palette


@pytest.fixture(scope="module")
def chemistry():
    return PatchedChemicalDatabase.from_chem_db(ChemicalDatabase.get_default())


@pytest.mark.parametrize("design", [False, True])
def test_palette_tracks_residue_identity_under_permutation(chemistry, design):
    names = {"ALA", "GLY", "AIB", "DALA", "ZN_tetrahedral"}
    blocks = [refined_from_raw(r) for r in chemistry.residues if r.name in names]
    assert {b.name for b in blocks} == names

    def choices(types):
        data = prepare_packer_palette(chemistry, types, design)
        result = {}
        for i, bt in enumerate(types):
            n = data.n_allowed[i]
            selected = data.allowed_types[i, :n]
            assert np.all(data.allowed_types[i, n:] == -1)
            assert data.is_original[i, :n].sum() == 1
            result[bt.name] = {
                (types[j].name, bool(data.repack_mask[i, k]))
                for k, j in enumerate(selected)
            }
        return result

    assert choices(blocks) == choices(blocks[::-1])


def test_empty_palette_has_typed_two_dimensional_choices(chemistry):
    data = prepare_packer_palette(chemistry, [])
    assert data.max_n_allowed == 0
    assert data.n_allowed.shape == (0,)
    assert data.n_allowed.dtype == np.int64
    assert data.allowed_types.shape == (0, 0)
    assert data.allowed_types.dtype == np.int64
    assert data.is_original.shape == data.repack_mask.shape == (0, 0)
    assert data.is_original.dtype == data.repack_mask.dtype == np.bool_
