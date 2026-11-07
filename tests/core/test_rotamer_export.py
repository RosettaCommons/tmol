"""Rotamer projection preserves source blocks independently of packed order."""

import numpy as np
import pytest

from tmol_core.chemical.default import default_patched_chemical_database
from tmol_core.pack.rotamer_set import atom_array_from_rotamers


def test_noncanonical_rotamers_keep_named_atoms_offsets_and_model_identity():
    residues = default_patched_chemical_database().residues
    types = [next(r for r in residues if r.name == name) for name in ("ALA", "DALA")]
    n_first, n_second = len(types[1].atoms), len(types[0].atoms)
    offsets = [2, n_first + 5]
    coords = np.full((offsets[1] + n_second + 3, 3), np.nan, np.float64)
    coords[offsets[0] : offsets[0] + n_first] = [1, 2, 3]
    coords[offsets[1] : offsets[1] + n_second] = [-4, 5, 6]
    original = coords.copy()
    atoms = atom_array_from_rotamers(
        types, coords, np.array([1, 0]), np.array([9, 2]), np.array(offsets)
    )
    assert len(atoms) == n_first + n_second
    assert atoms.get_annotation("model").dtype == np.int32
    np.testing.assert_array_equal(atoms.chain_id, "A")
    for model, name, residue, point in (
        (0, "DAL", 10, [1, 2, 3]),
        (1, "ALA", 3, [-4, 5, 6]),
    ):
        part = atoms[atoms.model == model]
        np.testing.assert_array_equal(part.res_name, name)
        np.testing.assert_array_equal(part.res_id, residue)
        np.testing.assert_array_equal(
            part.coord, np.broadcast_to(point, part.coord.shape)
        )
        assert {"N", "CA", "C", "O"} <= set(part.atom_name)
        np.testing.assert_array_equal(part.element[part.atom_name == "CA"], ["C"])
    np.testing.assert_array_equal(coords, original)


def test_empty_rotamers_preserve_biotite_error():
    with pytest.raises(IndexError, match="list index out of range"):
        atom_array_from_rotamers(
            [], np.empty((0, 3)), np.array([]), np.array([]), np.array([])
        )
