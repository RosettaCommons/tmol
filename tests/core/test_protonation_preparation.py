"""Custom atom-type donor capability must control the shared selection metadata."""

import attr
import numpy as np

from tmol_core.chemical import ChemicalDatabase
from tmol_core.chemical.patching import PatchedChemicalDatabase
from tmol_core.io.canonical_ordering import CanonicalOrdering
from tmol_core.io.protonation import prepare_protonation_plan


def test_custom_database_controls_donor_forms_without_stale_cached_chemistry():
    chemical = PatchedChemicalDatabase.from_chem_db(ChemicalDatabase.get_default())
    co = CanonicalOrdering.from_chemdb(chemical)
    types = np.array([[co.restype_io_equiv_classes.index("CYS"), -1]], np.int32)
    original = prepare_protonation_plan(co, chemical, types)
    assert original.donates.any()
    custom = attr.evolve(
        chemical,
        atom_types=tuple(
            attr.evolve(a, is_metal_donor=False) for a in chemical.atom_types
        ),
    )
    changed = prepare_protonation_plan(co, custom, types)
    assert not changed.donates.any()
    np.testing.assert_array_equal(changed.hydrogen, original.hydrogen)
    assert prepare_protonation_plan(co, chemical, types).donates.any()


def test_padding_never_requests_a_neutral_terminus_patch():
    from types import SimpleNamespace
    from tmol_core.chemical import NEUTRAL_TERMINUS_VAR_BASE
    from tmol_core.io.protonation import neutral_terminus_patches

    assert (
        neutral_terminus_patches(
            SimpleNamespace(restype_io_equiv_classes=["ALA"]),
            SimpleNamespace(variants=(), residues=()),
            np.array([[-1]], np.int32),
            np.array([[NEUTRAL_TERMINUS_VAR_BASE]], np.int32),
        )
        == ()
    )
