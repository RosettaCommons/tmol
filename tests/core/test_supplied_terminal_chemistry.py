"""Fixed-topology input only annotates chemistry actually supplied by the caller."""

from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest

from tmol_core.chemical import NEUTRAL_TERMINUS_VAR_BASE
from tmol_core.chemical.default import default_patched_chemical_database
from tmol_core.io import atom_array_from_file
from tmol_core.io.canonical_atom_array import prepare_canonical_atom_array
from tmol_core.io.canonical_ordering import CanonicalOrdering
from tmol_core.io.protonation_input import PROTONATION_VARIANT


@pytest.fixture(scope="module")
def chemistry():
    database = default_patched_chemical_database()
    return database, CanonicalOrdering.from_chemdb(database)


def _atoms():
    atoms = atom_array_from_file(
        Path(__file__).parents[2]
        / "tmol/tests/data/sweep_regressions/stated_terminus_5lh4.pdb"
    )
    atoms = atoms[atoms.res_id == 132]
    atoms.atom_name[atoms.atom_name == "H1"] = "H"
    return atoms


def _plan(atoms, chemistry):
    database, ordering = chemistry
    return prepare_canonical_atom_array(
        atoms,
        ordering,
        chemical_database=database,
        filter_missing_mainchain=False,
        missing_density_distance_threshold=0,
    )


def test_supplied_neutral_hydrogen_keeps_source_atoms_and_maps_to_terminal(chemistry):
    atoms = _atoms()
    original = atoms.copy()
    plan = _plan(atoms, chemistry)
    np.testing.assert_array_equal(plan.structure.coord, original.coord)
    np.testing.assert_array_equal(plan.structure.atom_name, original.atom_name)
    supplied = {tuple(row) for row in original.bonds.as_array()}
    actual = {tuple(row) for row in plan.structure.bonds.as_array()}
    nitrogen = int(np.flatnonzero(atoms.atom_name == "N")[0])
    hydrogen = int(np.flatnonzero(atoms.atom_name == "H")[0])
    expected_added = (*sorted((nitrogen, hydrogen)), int(struc.BondType.SINGLE))
    assert actual == supplied | {expected_added}
    np.testing.assert_array_equal(atoms.bonds.as_array(), original.bonds.as_array())
    assert PROTONATION_VARIANT not in atoms.get_annotation_categories()
    assert plan.fields["protonation_variants"][0, 0] >= NEUTRAL_TERMINUS_VAR_BASE
    hydrogen = int(np.flatnonzero(atoms.atom_name == "H")[0])
    slot = chemistry[1].restypes_atom_index_mapping["ALA"]["H1"]
    np.testing.assert_array_equal(
        plan.fields["coords"][0, 0, slot], atoms.coord[hydrogen]
    )


def test_missing_hydrogens_are_not_protonated_and_explicit_variants_survive(chemistry):
    atoms = _atoms()
    atoms = atoms[~np.isin(atoms.element, ("H", "D"))]
    plan = _plan(atoms, chemistry)
    assert plan.structure.array_length() == atoms.array_length()
    assert plan.fields["protonation_variants"] is None
    np.testing.assert_array_equal(plan.structure.coord, atoms.coord)
    atoms = _atoms()
    atoms.set_annotation(PROTONATION_VARIANT, np.full(len(atoms), -1, np.int8))
    plan = _plan(atoms, chemistry)
    np.testing.assert_array_equal(plan.fields["protonation_variants"], [[-1]])


def test_ensemble_requires_same_supplied_terminal_chemistry(chemistry):
    first = _atoms()
    second = first.copy()
    second.coord += 2
    plan = _plan(struc.stack([first, second]), chemistry)
    np.testing.assert_array_equal(
        plan.fields["protonation_variants"][0], plan.fields["protonation_variants"][1]
    )
    second.coord[np.isin(second.element, ("H", "D"))] = np.nan
    with pytest.raises(ValueError, match="Supplied terminal chemistry differs"):
        _plan(struc.stack([first, second]), chemistry)
