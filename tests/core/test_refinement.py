"""Chemical graph invariants across the complete shared residue corpus."""

import attr
import numpy as np
import networkx as nx
import pytest

from tmol_core.chemical import ChemicalDatabase
from tmol_core.chemical.patching import PatchedChemicalDatabase
from tmol_core.chemical.restypes import refined_from_raw


@pytest.fixture(scope="module")
def refined_types():
    database = PatchedChemicalDatabase.from_chem_db(ChemicalDatabase.get_default())
    return [refined_from_raw(raw) for raw in database.residues]


def test_all_residue_graph_distances_and_connection_identity(refined_types):
    for residue in refined_types:
        graph = nx.Graph()
        graph.add_nodes_from(range(residue.n_atoms))
        graph.add_edges_from(residue.bond_indices)
        expected = np.full((residue.n_atoms, residue.n_atoms), 6, np.int32)
        for source, distances in nx.all_pairs_shortest_path_length(graph, cutoff=6):
            for target, distance in distances.items():
                expected[source, target] = distance
        np.testing.assert_array_equal(residue.path_distance, expected)
        for conn, record in enumerate(residue.connections):
            atom = residue.atom_to_idx[record.atom]
            assert residue.ordered_connection_atoms[conn] == atom
            start, stop = residue.atom_all_bond_ranges[atom]
            assert [atom, -1, conn] in residue.all_bonds[start:stop].tolist()
        assert np.isfinite(residue.ideal_coords).all(), residue.name


def test_reordered_atoms_preserve_bond_chemistry(refined_types):
    # Metal, D-amino acid, and nucleotide records use the same refinement path.
    names = {residue.name for residue in refined_types}
    assert {"DALA", "DA"} <= names
    for original in refined_types:
        raw = attr.evolve(original, atoms=original.atoms[::-1])
        reordered = refined_from_raw(raw)
        for (a, b), order in original.bond_to_type.items():
            mapped = (
                reordered.atom_to_idx[original.atoms[a].name],
                reordered.atom_to_idx[original.atoms[b].name],
            )
            assert reordered.bond_to_type[mapped] == order
            assert reordered.bond_to_ringness[mapped] == original.bond_to_ringness[a, b]
        np.testing.assert_array_equal(reordered.ideal_coords, original.ideal_coords)

    from importlib.util import find_spec

    if find_spec("jax") is None:
        return

    import attrs
    from tmol_jax.chemical.restypes import _refined_from_raw
    from tmol_jax.database import ParameterDatabase

    raw = attrs.evolve(
        next(
            r
            for r in ParameterDatabase.get_default().chemical.residues
            if r.name == "ALA"
        )
    )
    first = _refined_from_raw(raw)
    expected = refined_from_raw(raw)
    first.ideal_coords[:] = 99
    first.path_distance[:] = 0
    first.atom_to_idx = {}
    first.icoors_index.clear()
    first.atom_names_set.clear()
    second = _refined_from_raw(raw)
    np.testing.assert_array_equal(second.ideal_coords, expected.ideal_coords)
    np.testing.assert_array_equal(second.path_distance, expected.path_distance)
    assert second.atom_to_idx == expected.atom_to_idx
    assert second.icoors_index == expected.icoors_index
    assert second.atom_names_set == expected.atom_names_set
    assert second.atoms is raw.atoms
    raw.icoors = tuple(
        attrs.evolve(icoor, d=icoor.d + 0.25) if icoor.name == "CB" else icoor
        for icoor in raw.icoors
    )
    changed = _refined_from_raw(raw)
    cb, ca = (changed.icoors_index[name] for name in ("CB", "CA"))
    old_length = np.linalg.norm(second.ideal_coords[cb] - second.ideal_coords[ca])
    new_length = np.linalg.norm(changed.ideal_coords[cb] - changed.ideal_coords[ca])
    np.testing.assert_allclose(new_length - old_length, 0.25, atol=1e-6)
    np.testing.assert_array_equal(second.ideal_coords, expected.ideal_coords)

    import attrs
    import gc
    import weakref
    from tmol_jax.chemical.restypes import _refined_from_raw
    from tmol_jax.database import ParameterDatabase

    raw = attrs.evolve(
        next(
            r
            for r in ParameterDatabase.get_default().chemical.residues
            if r.name == "ALA"
        ),
        name="CUSTOM",
    )
    reference = weakref.ref(raw)
    residue = _refined_from_raw(raw)
    del raw
    gc.collect()
    assert reference() is None
    assert residue.name == "CUSTOM"
