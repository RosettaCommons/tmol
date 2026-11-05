"""Count-pair weighting follows physical atom types, including mixed residues."""

from types import SimpleNamespace

import numpy as np

from tmol_core.scoring.count_pair import (
    electrostatic_distances,
    packed_ljlk_distances,
)


def test_mixed_types_and_charge_representatives_preserve_topology():
    # A linear five-atom topology with a canonical atom inside a ligand branch.
    paths = np.abs(np.arange(5)[:, None] - np.arange(5)).astype(np.int32)
    block = SimpleNamespace(
        atoms=[SimpleNamespace(atom_type=t) for t in ("CS", "CH2", "CS", "CS", "CS")],
        path_distance=paths,
        n_atoms=5,
    )
    original = paths.copy()
    distances, all_ligand = packed_ljlk_distances([block], 7, {"CH2"})
    assert distances[0, 0, 3] == 5  # generic/generic 1-4 pair: full weight
    assert distances[0, 0, 4] == 5  # generic/generic 1-5 pair: full weight
    assert distances[0, 1, 4] == 3  # canonical/generic pair: canonical rule
    assert distances[0, 0, 2] == 2  # short-range exclusions are unchanged
    assert (distances[0, 5:, :] == 6).all()
    assert all_ligand.tolist() == [0]
    reps = np.array([0, 1, 2, 0, 4])
    inter, intra = electrostatic_distances(block, reps, {"CH2"})
    assert inter[4, 3] == 4
    assert intra[4, 3] == 5
    assert intra[1, 4] == 3
    np.testing.assert_array_equal(paths, original)
