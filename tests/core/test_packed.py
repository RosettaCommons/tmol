"""Packed host metadata preserves physical identities and unused storage."""

import attr
import numpy as np
import pytest

from tmol_core.chemical import ChemicalDatabase
from tmol_core.chemical.patching import PatchedChemicalDatabase
from tmol_core.chemical.restypes import refined_from_raw
from tmol_core.chemical.packed import PackedBlockTypesData


@pytest.fixture(scope="module")
def chemistry():
    return PatchedChemicalDatabase.from_chem_db(ChemicalDatabase.get_default())


def test_packed_atom_order_and_physical_connections(chemistry):
    # Reverse every atom list: storage must follow atom identity, not icoor order.
    types = [
        refined_from_raw(attr.evolve(raw, atoms=raw.atoms[::-1]))
        for raw in chemistry.residues
    ]
    data = PackedBlockTypesData.from_restypes(chemistry, types)
    for i, bt in enumerate(types):
        n = bt.n_atoms
        assert data.atom_names[i][:n] == tuple(a.name for a in bt.atoms)
        for atom, record in enumerate(bt.atoms):
            assert (
                chemistry.atom_types[data.atom_types[i, atom]].name == record.atom_type
            )
            expected = bt.ideal_coords[bt.icoors_index[record.name]]
            np.testing.assert_allclose(
                data.ideal_coords[i, atom], expected, rtol=1e-6, atol=1e-6
            )
        for connection, record in enumerate(bt.connections):
            assert bt.atoms[data.conn_atom[i, connection]].name == record.atom
            assert data.kinematic_conn[i, connection] == record.kinematic
        assert not data.atom_is_real[i, n:].any()
        assert (data.conn_atom[i, len(bt.connections) :] == -1).all()
        assert not data.kinematic_conn[i, len(bt.connections) :].any()
        assert (data.bond_indices[i, len(bt.bond_indices) :] == -1).all()
        np.testing.assert_array_equal(
            data.intrares_path_distance[i, :n, :n], bt.path_distance
        )


def test_empty_palette_retains_requested_connection_capacity(chemistry):
    data = PackedBlockTypesData.from_restypes(chemistry, [], min_n_conn=2)
    assert data.conn_atom.shape == (0, 2)
    assert data.kinematic_conn.shape == (0, 2)
    assert data.atom_paths_from_conn.shape == (0, 2, 13, 3)
    assert data.torsion_uaids.shape == (0, 0, 4, 3)
    assert data.atom_is_hydrogen.shape == (0, 0)
    assert data.ideal_coords.shape == (0, 0, 3)


def test_missing_atom_type_fails_before_device_transfer(chemistry):
    raw = chemistry.residues[0]
    residue = refined_from_raw(
        attr.evolve(
            raw,
            atoms=(attr.evolve(raw.atoms[0], atom_type="missing-type"), *raw.atoms[1:]),
        )
    )
    with pytest.raises(KeyError, match="missing-type"):
        PackedBlockTypesData.from_restypes(chemistry, [residue])
