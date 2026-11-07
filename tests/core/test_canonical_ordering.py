"""Canonical chemistry remains valid for subsets without protein special cases."""

import attr
import numpy as np
import pytest

from tmol_core.chemical import ChemicalDatabase
from tmol_core.chemical.patching import PatchedChemicalDatabase
from tmol_core.io.canonical_ordering import CanonicalOrdering


@pytest.fixture(scope="module")
def ordering():
    chemistry = PatchedChemicalDatabase.from_chem_db(ChemicalDatabase.get_default())
    keep = {"GLY", "DALA", "DA", "HOH", "ZN_tetrahedral"}
    chemistry = attr.evolve(
        chemistry, residues=tuple(r for r in chemistry.residues if r.base_name in keep)
    )
    return CanonicalOrdering.from_chemdb(chemistry)


def test_mixed_subset_has_no_invented_histidine_or_cysteine(ordering):
    assert ordering.his_inds.his_co_aa_inds == ()
    assert ordering.cys_inds.cys_co_aa_inds == ()
    assert {"GLY", "DAL", "DA", "HOH"} <= set(ordering.restype_io_equiv_classes)
    # DNA's 5-prime variant can omit phosphate, so it is not required input.
    assert "P" not in ordering.restypes_required_mainchain_atoms["DA"]


def test_source_atom_padding_and_invalid_names(ordering):
    residues, atoms, real = ordering.create_src_2_tmol_mappings(
        np.array(["GLY", "MISSING"]), {"GLY": ["CA", ""], "MISSING": ["X", ""]}
    )
    np.testing.assert_array_equal(
        residues, [ordering.restype_io_equiv_classes.index("GLY"), -1]
    )
    assert atoms[0, 0] == ordering.restypes_atom_index_mapping["GLY"]["CA"]
    np.testing.assert_array_equal(atoms[:, 1], [-1, -1])
    np.testing.assert_array_equal(real, [[True, False], [False, False]])
    with pytest.raises(ValueError, match="atom NOT_AN_ATOM not in"):
        ordering.create_src_2_tmol_mappings(["GLY"], {"GLY": ["NOT_AN_ATOM"]})


def test_empty_source_mapping_retains_shapes_and_dtypes(ordering):
    residues, atoms, real = ordering.create_src_2_tmol_mappings([], {})
    assert residues.shape == (0,)
    assert atoms.shape == real.shape == (0, 0)
    assert residues.dtype == atoms.dtype == np.int64
    assert real.dtype == np.bool_
