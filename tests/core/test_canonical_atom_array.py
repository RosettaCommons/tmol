"""Canonical host plans preserve source identities and fixed-topology semantics."""

import biotite.structure as struc
import numpy as np
import pytest

from tmol_core.chemical.default import default_patched_chemical_database
from tmol_core.io.canonical_atom_array import prepare_canonical_atom_array
from tmol_core.io.canonical_ordering import CanonicalOrdering
from tmol_core.io.input_termini import EXPLICIT_TERMINI


@pytest.fixture(scope="module")
def ordering():
    return CanonicalOrdering.from_chemdb(default_patched_chemical_database())


def _atoms():
    atoms = struc.AtomArray(9)
    atoms.res_name = np.array(["GLY"] * 4 + ["DAL"] * 5)
    atoms.atom_name = np.array(["N", "CA", "C", "O"] * 2 + ["CB"])
    atoms.element = np.array(["N", "C", "C", "O"] * 2 + ["C"])
    atoms.res_id = np.array([18] * 4 + [19] * 5)
    atoms.chain_id[:] = "B"
    atoms.ins_code[:] = "A"
    atoms.coord = np.arange(27, dtype=np.float32).reshape(9, 3) / 10
    atoms.set_annotation("occupancy", np.arange(9, dtype=np.float32) / 10)
    atoms.set_annotation(EXPLICIT_TERMINI, np.array([0] * 4 + [1] * 5))
    return atoms


def test_batched_metadata_and_stated_ports_are_independent(ordering):
    atoms = _atoms()
    shifted = atoms.copy()
    shifted.coord += 3
    stack = struc.stack([atoms, shifted])
    source_coords = stack.coord.copy()
    plan = prepare_canonical_atom_array(
        stack, ordering, missing_density_distance_threshold=0
    )
    fields = plan.fields
    np.testing.assert_array_equal(fields["res_labels"], [[18, 19], [18, 19]])
    np.testing.assert_array_equal(fields["chain_id"], np.zeros((2, 2), np.int32))
    np.testing.assert_array_equal(
        fields["res_not_connected"], [[[False, True], [True, False]]] * 2
    )
    slot = ordering.restypes_atom_index_mapping["DAL"]["CB"]
    np.testing.assert_array_equal(fields["coords"][:, 1, slot], source_coords[:, 8])
    np.testing.assert_array_equal(
        fields["atom_occupancy"][:, 1, slot], np.repeat(atoms.occupancy[8], 2)
    )
    fields["coords"][0, 1, slot] = -10
    fields["res_labels"][0, 0] = -10
    assert fields["res_labels"][1, 0] == 18
    np.testing.assert_array_equal(stack.coord, source_coords)


def test_external_coordinates_preserve_topology_despite_missing_reference(ordering):
    atoms = _atoms()
    atoms.coord[:] = np.nan
    plan = prepare_canonical_atom_array(atoms, ordering, atom37_n_poses=3)
    assert plan.fields["coords"] is None
    assert plan.fields["res_types"].shape == (3, 2)
    assert plan.valid_atom_mask.all()
    assert plan.fields["res_not_connected"][:, 0, 1].all()
    assert plan.fields["res_not_connected"][:, 1, 0].all()
