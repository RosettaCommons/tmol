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
